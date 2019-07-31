import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torchvision.models.resnet import resnet18


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def fix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1:
            m.weight.requires_grad = False
            if m.bias is not None:
                m.bias.requires_grad = False

    net.apply(fix_func)


def unfix_grad(net):
    def fix_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('BatchNorm2d') != -1 or classname.find('Linear') != -1:
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True

    net.apply(fix_func)


def define_TrecgNet(cfg, upsample=None, device=None):

    if upsample is None:
        upsample = not cfg.NO_UPSAMPLE

    model = TRecgNet_Upsample_Resiual(cfg, encoder=cfg.ARCH, upsample=upsample, device=device)

    return model

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv_norm_relu(dim_in, dim_out, kernel_size=3, norm=nn.BatchNorm2d, stride=1, padding=1,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]


##############################################################################
# Moduels
##############################################################################
class Upsample_Interpolate(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=1, padding=0, norm=nn.BatchNorm2d, scale=2, mode='bilinear', activate=True):
        super(Upsample_Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = norm(dim_out)

    def forward(self, x, activate=True):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        conv_out = self.conv(x)
        conv_out = self.bn(conv_out)
        if activate:
            conv_out = nn.ReLU(True)(conv_out)
        return x, conv_out


class UpsampleBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear', upsample=True):
        super(UpsampleBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        if upsample:
            if inplanes != planes:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 3, 1

            self.upsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                norm(planes))
        else:
            self.upsample = None

        self.scale = scale
        self.mode = mode

    def forward(self, x):

        if self.upsample is not None:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
            residual = self.upsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

##############################################################################
# Translate to recognize
##############################################################################
class Content_Model(nn.Module):

    def __init__(self, cfg, criterion=None):
        super(Content_Model, self).__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.net = cfg.WHICH_CONTENT_NET

        if 'resnet' in self.net:
            from .pretrained_resnet import ResNet
            self.model = ResNet(self.net, cfg)

        fix_grad(self.model)
        # print_network(self)

    def forward(self, x, in_channel=3, layers=None):

        self.model.eval()

        if layers is None:
            layers = self.cfg.CONTENT_LAYERS.split(',')

        layer_wise_features = self.model(x, layers)
        return layer_wise_features


class TRecgNet_Upsample_Resiual(nn.Module):

    def __init__(self, cfg, encoder='resnet18', upsample=True,  device=None):
        super(TRecgNet_Upsample_Resiual, self).__init__()

        self.encoder = encoder
        self.cfg = cfg
        self.upsample = upsample
        self.dim_noise = 128
        self.device = device
        self.avg_pool_size = 14

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            # places model downloaded from http://places2.csail.mit.edu/
            checkpoint = torch.load(self.cfg.CONTENT_MODEL_PATH, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            resnet.load_state_dict(state_dict)
            print('place resnet18 loaded....')
        else:
            resnet = resnet18(pretrained=pretrained)
            print('{0} pretrained:{1}'.format(encoder, str(pretrained)))

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

        self.build_upsample_layers(dims)

        self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        self.fc = nn.Linear(dims[4], cfg.NUM_CLASSES)

        if pretrained and upsample:

            init_weights(self.up1, 'normal')
            init_weights(self.up2, 'normal')
            init_weights(self.up3, 'normal')
            init_weights(self.up4, 'normal')
            init_weights(self.skip_3, 'normal')
            init_weights(self.skip_2, 'normal')
            init_weights(self.skip_1, 'normal')
            init_weights(self.up_image, 'normal')

        elif not pretrained:

            init_weights(self, 'normal')

    def _make_upsample(self, block, planes, norm=nn.BatchNorm2d, is_upsample=False):

        upsample = None
        if self.inplanes != planes or is_upsample:
            upsample = Upsample_Interpolate(self.inplanes, planes, kernel_size=1, padding=0, norm=norm, activate=False)

        layer = block(self.inplanes, planes, norm=norm, upsample=upsample)
        self.inplanes = planes

        return layer

    def build_upsample_layers(self, dims):

        # norm = nn.BatchNorm2d
        norm = nn.InstanceNorm2d

        # self.inplanes = dims[4] + self.dim_noise if self.use_noise else dims[4]
        self.up1 = UpsampleBasicBlock(dims[4], dims[3], kernel_size=1, padding=0, norm=norm)
        self.up2 = UpsampleBasicBlock(dims[3], dims[2], kernel_size=1, padding=0, norm=norm)
        self.up3 = UpsampleBasicBlock(dims[2], dims[1], kernel_size=1, padding=0, norm=norm)
        self.up4 = UpsampleBasicBlock(dims[1], dims[1], kernel_size=3, padding=1, norm=norm)

        self.skip_3 = conv_norm_relu(dims[3], dims[3], kernel_size=1, padding=0, norm=norm)
        self.skip_2 = conv_norm_relu(dims[2], dims[2], kernel_size=1, padding=0, norm=norm)
        self.skip_1 = conv_norm_relu(dims[1], dims[1], kernel_size=1, padding=0, norm=norm)

        self.up_image = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, out_keys=None, phase='train', content_layers=None, return_losses=True):
        out = {}

        out['0'] = self.relu(self.bn1(self.conv1(source)))
        out['1'] = self.layer1(out['0'])
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])

        if self.upsample and 'gen_img' in out_keys:
            skip1 = self.skip_1(out['1'])  # 64 / 128
            skip2 = self.skip_2(out['2'])  # 128 / 256
            skip3 = self.skip_3(out['3'])  # 256 / 512

            upconv4 = self.up1(out['4'])
            upconv3 = self.up2(upconv4 + skip3)
            upconv2 = self.up3(upconv3 + skip2)
            upconv1 = self.up4(upconv2 + skip1)

            out['gen_img'] = self.up_image(upconv1)

        out['avgpool'] = self.avgpool(out['4'])
        avgpool = out['avgpool'].view(source.size(0), -1)
        out['cls'] = self.fc(avgpool)

        result = []
        for key in out_keys:
            if isinstance(key, list):
                item = [out[subkey] for subkey in key]
            else:
                item = out[key]
            result.append(item)

        return result


class Fusion(nn.Module):

    def __init__(self, cfg, rgb_model=None, depth_model=None, device='cuda'):
        super(Fusion, self).__init__()
        self.cfg = cfg
        self.device = device
        self.rgb_model = rgb_model
        self.depth_model = depth_model
        self.net_RGB = self.construct_single_modal_net(rgb_model)
        self.net_depth = self.construct_single_modal_net(depth_model)

        if cfg.FIX_GRAD:
            fix_grad(self.net_RGB)
            fix_grad(self.net_depth)

        self.avgpool = nn.AvgPool2d(14, 1)
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, cfg.NUM_CLASSES)
        )

        init_weights(self.fc, 'normal')

    # only keep the classification branch
    def construct_single_modal_net(self, model):
        if isinstance(model, nn.DataParallel):
            model = model.module

        ops = [model.conv1, model.bn1, model.relu, model.layer1, model.layer2,
                                   model.layer3, model.layer4]
        return nn.Sequential(*ops)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def forward(self, input_rgb, input_depth, label, out_keys=None):

        out = {}

        rgb_specific = self.net_RGB(input_rgb)
        depth_specific = self.net_depth(input_depth)

        concat = torch.cat((rgb_specific, depth_specific), 1).to(self.device)
        x = self.avgpool(concat)
        x = x.view(x.size(0), -1)
        out['cls'] = self.fc(x)

        result = []
        for key in out_keys:
            if isinstance(key, list):
                item = [out[subkey] for subkey in key]
            else:
                item = out[key]
            result.append(item)

        return result

