# Copyright 2023 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as thf
import torchvision


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':   
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        
        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        
    def forward(self, x):
        x = self.conv(self.pad(x)) 
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        
        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):   
        residual = x
        out = self.model(x)
        out += residual
        return out

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        
    def forward(self, x):
        return self.model(x)
        

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'




class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
            
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        
        x = (x - mean) / (std + self.eps)
            
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
        



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, activ='relu', norm='none'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        # initialize activation
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'lrelu':
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        elif activ == 'selu':
            self.activ = nn.SELU(inplace=True)
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'silu':
            self.activ = nn.SiLU(inplace=True)
        elif activ == 'none':
            self.activ = None
        else:
            assert 0, "Unsupported activ: {}".format(activ)
        
        # initialize normalization
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activ:
            x = self.activ(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, in_channels, skip_channels='default', out_channels='default', activ='relu', norm='none'):
        super().__init__()
        if skip_channels == 'default':
            skip_channels = in_channels//2
        if out_channels == 'default':
            out_channels = in_channels//2
        self.up = nn.Upsample(scale_factor=(2,2))
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = Conv2d(in_channels, out_channels, 2, 1, 0, activ=activ, norm=norm)
        
        self.conv2 = Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, activ=activ, norm=norm)
    
    def forward(self, x, skip):
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


class Secret2Image(nn.Module):
    def __init__(self, resolution, secret_len):
        super().__init__()
        assert resolution % 16 == 0, "Resolution must be a multiple of 16."
        self.dense = nn.Linear(secret_len, 16*16*3)
        self.upsample = nn.Upsample(scale_factor=(resolution//16, resolution//16))
        self.activ = nn.ReLU(inplace=True)
    
    def forward(self, secret):
        x = self.dense(secret)
        x = x.view((-1, 3, 16, 16))
        x = self.activ(self.upsample(x))
        return x


class Unet1(nn.Module):
    def __init__(self, resolution=256, secret_len=100, width=32, ndown=4, nmiddle=0, activ='relu'):
        super().__init__()
        self.secret_len = secret_len
        self.ndown = ndown
        self.resolution = resolution
        self.secret2image = Secret2Image(resolution, secret_len)
        self.pre = Conv2d(6, width, 3, 1, 1, activ=activ, norm='none')

        self.enc = nn.ModuleList()
        ch = width
        for _ in range(ndown):
            self.enc.append(Conv2d(ch, ch*2, 3, 2, 1, norm='none'))
            ch *= 2

        if nmiddle > 0:
            self.middle = nn.ModuleList()
            for _ in range(nmiddle):
                self.middle.append(ResBlocks(nmiddle, ch, norm='none', activation=activ, pad_type='zero'))
        else:
            self.middle = None

        self.dec = nn.ModuleList()
        for i in range(ndown):
            skip_width = ch//2 if i < ndown-1 else ch//2+6
            self.dec.append(DecBlock(ch, skip_width, activ=activ, norm='none'))
            ch //= 2
        
        self.post = nn.Sequential(
            Conv2d(ch, ch, 3, 1, 1, activ=activ, norm='none'),
            Conv2d(ch, ch//2, 1, 1, 0, activ='silu', norm='none'),
            Conv2d(ch//2, 3, 1, 1, 0, activ='tanh', norm='none')
        )
    
    def forward(self, image, secret=None):
        if secret is None:
            secret = torch.randn(image.shape[0], self.secret_len, device=image.device)
        secret = self.secret2image(secret)
        inputs = torch.cat([image, secret], dim=1)
        enc = []
        x = self.pre(inputs)
        for block in self.enc:
            enc.append(x)
            x = block(x)
        enc = enc[::-1]
        if self.middle:
            for block in self.middle:
                x = block(x)
        for i, (block, skip) in enumerate(zip(self.dec, enc)):
            if i < self.ndown-1:
                x = block(x, skip)
            else:
                x = block(x, torch.cat([skip, inputs], dim=1))
        x = self.post(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, resolution=224, secret_len=100):
        super().__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = 3
        self.decoder = nn.Sequential(
            nn.Conv2d(self.IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # resolution / 2
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # resolution / 4
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # resolution / 8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # resolution / 16
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),  # resolution / 32
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, secret_len),
        )

    def forward(self, image, **kwargs):
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)
    

class MSResNet(nn.Module):
    def __init__(self, input_dim=3, dim=64, norm='none', activ='lrelu', n_layer=4, gan_type='lsgan', num_scales=3, pad_type='reflect', out_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.norm = norm
        self.activ = activ
        self.n_layer = n_layer
        self.gan_type = gan_type
        self.num_scales = num_scales
        self.pad_type = pad_type
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.out_dim = out_dim
        self.cnns = nn.ModuleList()
        for i in range(num_scales):
            d = out_dim // (2**(num_scales - i - 1)) if out_dim > 1 else 1
            self.cnns.append(self._make_net(d))
            
    
    def _make_net(self, out_dim):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, out_dim, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x
    
    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs


class SecretDecoder(nn.Module):
    def __init__(self, arch='resnet18', resolution=224, secret_len=100):
        super().__init__()
        self.resolution = resolution
        self.arch = arch
        if arch == 'resnet18':
#            self.decoder = torchvision.models.resnet18(pretrained=True, progress=False)
            self.decoder = torchvision.models.resnet18()
            self.decoder.fc = nn.Linear(self.decoder.fc.in_features, secret_len)
        elif arch == 'resnet50':
            self.decoder = torchvision.models.resnet50()
#            self.decoder = torchvision.models.resnet50(pretrained=True, progress=False)
            self.decoder.fc = nn.Linear(self.decoder.fc.in_features, secret_len)
        elif arch == 'resnext50':
            self.decoder = torchvision.models.resnext50_32x4d(weights='ResNet50_Weights.IMAGENET1K_V1', progress=False)
            self.decoder.fc = nn.Linear(self.decoder.fc.in_features, secret_len)
        elif arch == 'googlenet':
            self.decoder = torchvision.models.googlenet(pretrained=True, progress=False)
            self.decoder.fc = nn.Linear(self.decoder.fc.in_features, secret_len)
        elif arch == 'densenet121':
            self.decoder = torchvision.models.densenet121(pretrained=True, progress=False)
            self.decoder.classifier = nn.Linear(self.decoder.classifier.in_features, secret_len)
        elif arch == 'efficientnet':
            self.decoder = torchvision.models.efficientnet_b0(pretrained=True, progress=False)
            self.decoder.classifier[1] = nn.Linear(self.decoder.classifier[1].in_features, secret_len)
        elif arch == 'regnet':
            self.decoder = torchvision.models.regnet_y_1_6gf(pretrained=True, progress=False)
            self.decoder.fc = nn.Linear(self.decoder.fc.in_features, secret_len)
        elif arch == 'convnext':
            self.decoder = torchvision.models.convnext_tiny(pretrained=True, progress=False)
            self.decoder.classifier[2] = nn.Linear(self.decoder.classifier[2].in_features, secret_len)
        elif arch == 'vgg':
            self.decoder = torchvision.models.vgg16(pretrained=True, progress=False)
            self.decoder.classifier[6] = nn.Linear(self.decoder.classifier[6].in_features, secret_len)
        elif arch == 'simple':
            self.decoder = SimpleCNN(resolution, secret_len)
        elif arch == 'msresnet':
            self.decoder = MSResNet(resolution, secret_len)
        else:
            raise ValueError('Unknown architecture')
        
    def forward(self, image, **kwargs):
        if self.arch in ['resnet50', 'resnet18'] and min(image.shape[-2:]) > self.resolution:
            image = thf.interpolate(image, size=(self.resolution, self.resolution), mode='bilinear', align_corners=False)
        x = self.decoder(image)
        return x
