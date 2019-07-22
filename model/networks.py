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

def define_TrecgNet(cfg, use_noise, upsample=None, device=None):
    if upsample is None:
        upsample = not cfg.NO_UPSAMPLE

    model = TRecgNet_Upsample_Resiual(cfg, encoder=cfg.ARCH, upsample=upsample,
                        use_noise=use_noise, device=device)

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


class UpBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm, upsample=None):
        super(UpBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm(planes)

        self.upsample = upsample

    def forward(self, x):

        residual = x
        if self.upsample is not None:
            x, conv_out = self.upsample(x, activate=False)
            residual = conv_out

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

    def __init__(self, cfg, encoder='resnet18', upsample=True,
                 use_noise=False, device=None):
        super(TRecgNet_Upsample_Resiual, self).__init__()

        self.encoder = encoder
        self.cfg = cfg
        self.upsample = upsample
        self.dim_noise = 128
        self.use_noise = use_noise
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

        self.inplanes = dims[4] + self.dim_noise if self.use_noise else dims[4]
        self.up1 = Upsample_Interpolate(self.inplanes, dims[3], kernel_size=1, padding=0, norm=norm, activate=True)

        self.inplanes = dims[3]
        self.up2 = self._make_upsample(UpBasicBlock, 128, norm=norm, is_upsample=True)
        self.up3 = self._make_upsample(UpBasicBlock, 64, norm=norm, is_upsample=True)
        self.up4 = self._make_upsample(UpBasicBlock, 64, norm=norm, is_upsample=True)

        skip3_channel = dims[3]
        skip2_channel = dims[2]
        skip1_channel = dims[1]
        self.skip_3 = conv_norm_relu(skip3_channel, skip3_channel, kernel_size=1, padding=0, norm=norm)
        self.skip_2 = conv_norm_relu(skip2_channel, skip2_channel, kernel_size=1, padding=0, norm=norm)
        self.skip_1 = conv_norm_relu(skip1_channel, skip1_channel, kernel_size=1, padding=0, norm=norm)

        self.up_image = nn.Sequential(
            conv_norm_relu(64, 64, kernel_size=3, padding=1, norm=norm),
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

            if self.use_noise:
                noise = torch.FloatTensor(source.size(0), self.dim_noise, self.avg_pool_size,
                                          self.avg_pool_size).normal_(0, 1).to(self.device)
                _, upconv4 = self.up1(torch.cat((out['4'], noise), 1))
            else:
                _, upconv4 = self.up1(out['4'], activate=True)

            skip1 = self.skip_1(out['1'])  # 64 / 128
            skip2 = self.skip_2(out['2'])  # 128 / 256
            skip3 = self.skip_3(out['3'])  # 256 / 512

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

