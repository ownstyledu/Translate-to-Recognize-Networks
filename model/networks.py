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
    print(net.__class__.__name__)
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

def define_TrecgNet(cfg, use_noise=None, upsample=None, device=None):
    if upsample is None:
        upsample = not cfg.NO_UPSAMPLE

    if 'resnet' in cfg.ARCH:
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

def conv_norm_relu(dim_in, dim_out, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d,
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

def expand_Conv(module, in_channels):
    def expand_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.in_channels = in_channels
            m.out_channels = m.out_channels
            mean_weight = torch.mean(m.weight, dim=1, keepdim=True)
            m.weight.data = mean_weight.repeat(1, in_channels, 1, 1).data

    module.apply(expand_func)


##############################################################################
# Moduels
##############################################################################
class Upsample_Interpolate(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear', reduce_dim=False):
        super(Upsample_Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode
        if reduce_dim:
            dim_out = int(dim_out / 2)
            self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=1, stride=1, padding=0, norm=norm)
            self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_in, kernel_size=3, stride=1, padding=1, norm=norm)
        else:
            self.conv_norm_relu1 = conv_norm_relu(dim_in, dim_out, kernel_size=kernel_size, stride=1, padding=padding, norm=norm)
            self.conv_norm_relu2 = conv_norm_relu(dim_out, dim_out, kernel_size=3, stride=1, padding=1, norm=norm)

    def forward(self, x, activate=True):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
        x = self.conv_norm_relu1(x)
        x = self.conv_norm_relu2(x)
        return x

class Upconv_ConvTransposed(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, output_padding=1):
        super(Upconv_ConvTransposed, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


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
#########################################
class Discriminator(nn.Module):
    # initializers
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        # norm = nn.BatchNorm2d
        self.d_downsample_num = 4
        norm = nn.InstanceNorm2d
        convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1),
            norm(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1),
            norm(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, 3, 1, 1),
            norm(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 1, 3, 2, 1)
        )

        if self.cfg.NO_LSGAN:
            convs.add_module('sigmoid', nn.Sigmoid())

        self.model = nn.Sequential(*convs)
        init_weights(self.model, 'normal')

    def forward(self, x):
        return self.model(x)

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

    def forward(self, x, target, in_channel=3, layers=None):

        # important when set content_model as the attr of trecg_net
        self.model.eval()

        layers = layers
        if layers is None or not layers:
            layers = self.cfg.CONTENT_LAYERS.split(',')

        input_features = self.model((x + 1) / 2, layers)
        target_targets = self.model((target + 1) / 2, layers)
        len_layers = len(layers)
        loss_fns = [self.criterion] * len_layers
        alpha = [1] * len_layers

        content_losses = [alpha[i] * loss_fns[i](gen_content, target_targets[i])
                          for i, gen_content in enumerate(input_features)]
        loss = sum(content_losses) * self.cfg.ALPHA_CONTENT
        return loss


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
        self.avg_pool_size = 7

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        fc_input_nc = dims[4] if encoder == 'resnet18' else dims[6]

        if cfg.PRETRAINED == 'imagenet' or cfg.PRETRAINED == 'place':
            pretrained = True
        else:
            pretrained = False

        if cfg.PRETRAINED == 'place':
            resnet = models.__dict__['resnet18'](num_classes=365)
            load_path = "/home/dudapeng/workspace/pretrained/resnet18_places365.pth"
            checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
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

        if self.upsample:
            self.avg_pool_size = 14
            self.build_upsample_layers(dims)

        self.avgpool = nn.AvgPool2d(self.avg_pool_size, 1)
        self.fc = nn.Linear(fc_input_nc, cfg.NUM_CLASSES)

        if pretrained and upsample:

            init_weights(self.deconv1, 'normal')
            init_weights(self.deconv2, 'normal')
            init_weights(self.deconv3, 'normal')
            init_weights(self.deconv4, 'normal')
            init_weights(self.skip_3, 'normal')
            init_weights(self.skip_2, 'normal')
            init_weights(self.skip_1, 'normal')
            init_weights(self.up_image, 'normal')

        elif not pretrained:

            init_weights(self, 'normal')

    def set_content_model(self, content_model):
        self.content_model = content_model

    def set_pix2pix_criterion(self, criterion):
        self.pix2pix_criterion = criterion.to(self.device)

    def set_cls_criterion(self, criterion):
        self.cls_criterion = criterion.to(self.device)

    def build_upsample_layers(self, dims):

        norm = nn.InstanceNorm2d

        self.deconv1 = Upsample_Interpolate(dims[4], dims[3], kernel_size=1, padding=0, norm=norm)
        self.deconv2 = Upsample_Interpolate(dims[3], dims[2], kernel_size=1, padding=0, norm=norm)
        self.deconv3 = Upsample_Interpolate(dims[2], dims[1], kernel_size=1, padding=0, norm=norm)
        self.deconv4 = Upsample_Interpolate(dims[1], dims[1], kernel_size=3, padding=1, norm=norm)

        skip3_channel = dims[3]
        skip2_channel = dims[2]
        skip1_channel = dims[1]
        self.skip_3 = conv_norm_relu(skip3_channel, skip3_channel, kernel_size=1, padding=0, norm=norm)
        self.skip_2 = conv_norm_relu(skip2_channel, skip2_channel, kernel_size=1, padding=0, norm=norm)
        self.skip_1 = conv_norm_relu(skip1_channel, skip1_channel, kernel_size=1, padding=0, norm=norm)

        self.up_image = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, source=None, target=None, label=None, out_keys=None, phase='train', content_layers=None, return_losses=True):
        out = {}

        if self.cfg.FIVE_CROP and phase == 'test':

            self.bs, self.ncrops, c, h, w = source.size()
            source = source.view(-1, c, h, w)

        if self.upsample:
            out['0'] = self.relu(self.bn1(self.conv1(source)))
        else:
            out['0'] = self.maxpool(self.relu(self.bn1(self.conv1(source))))

        out['1'] = self.layer1(out['0'])
        out['2'] = self.layer2(out['1'])
        out['3'] = self.layer3(out['2'])
        out['4'] = self.layer4(out['3'])

        if self.upsample and 'gen_img' in out_keys:

            skip1 = out['1']
            skip2 = out['2']
            skip3 = out['3']

            upconv4 = self.deconv1(out['4'])
            upconv3 = self.deconv2(upconv4 + skip3)
            upconv2 = self.deconv3(upconv3 + skip2)
            upconv1 = self.deconv4(upconv2 + skip1)

            out['gen_img'] = self.up_image(upconv1)


        out['avgpool'] = self.avgpool(out['4'])
        avgpool = out['avgpool'].view(source.size(0), -1)
        out['cls'] = self.fc(avgpool)

        loss_content = None
        loss_cls = None
        loss_pix2pix = None

        if return_losses and not self.cfg.INFERENCE:

            if 'PIX2PIX' in self.cfg.LOSS_TYPES and target is not None and phase == 'train':
                loss_pix2pix = self.pix2pix_criterion(out['gen_img'], target) * self.cfg.ALPHA_PIX2PIX

            if 'SEMANTIC' in self.cfg.LOSS_TYPES and target is not None and phase == 'train':

                loss_content = self.content_model(out['gen_img'], target, layers=content_layers) * self.cfg.ALPHA_CLS

            if 'CLS' in self.cfg.LOSS_TYPES and not self.cfg.UNLABELED:

                if self.cfg.FIVE_CROP and phase == 'test':
                    out['cls'] = out['cls'].view(self.bs, self.ncrops, -1).mean(1)

                loss_cls = self.cls_criterion(out['cls'], label) * self.cfg.ALPHA_CLS

        result = []
        for key in out_keys:
            if isinstance(key, list):
                item = [out[subkey] for subkey in key]
            else:
                item = out[key]
            result.append(item)

        return result, {'cls_loss': loss_cls, 'content_loss': loss_content, 'pix2pix_loss': loss_pix2pix}