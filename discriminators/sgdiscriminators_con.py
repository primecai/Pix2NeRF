import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def mk_conv2d(*args, sn=False, **kwargs):
    m = nn.Conv2d(*args, **kwargs)
    if sn:
        m = spectral_norm(m)
    return m


def mk_linear(*args, sn=False, **kwargs):
    m = nn.Linear(*args, **kwargs)
    if sn:
        m = spectral_norm(m)
    return m


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean([2, 3])


class AdapterBlock(nn.Module):
    def __init__(self, output_channels, sn=False):
        super().__init__()
        self.model = nn.Sequential(
            mk_conv2d(3, output_channels, 1, padding=0, sn=sn),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.model(input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class StridedResidualConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, sn=False):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(
            mk_conv2d(inplanes, planes, kernel_size=kernel_size, padding=p, sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
            mk_conv2d(planes, planes, kernel_size=kernel_size, stride=2, padding=p, sn=sn),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)
        self.proj = mk_conv2d(inplanes, planes, 1, stride=2, sn=sn)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity)/math.sqrt(2)
        return y


class StridedDiscriminator(nn.Module):
    def __init__(self, sn=False, **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([
            StridedResidualConvBlock(32, 64, sn=sn), # 6 256x256 -> 128x128
            StridedResidualConvBlock(64, 128, sn=sn), # 5 128x128 -> 64x64
            StridedResidualConvBlock(128, 256, sn=sn), # 4 64x64 -> 32x32
            StridedResidualConvBlock(256, 400, sn=sn), # 3 32x32 -> 16x16
            StridedResidualConvBlock(400, 400, sn=sn), # 2 16x16 -> 8x8
            StridedResidualConvBlock(400, 400, sn=sn), # 1 8x8 -> 4x4
            StridedResidualConvBlock(400, 400, sn=sn), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList([
            AdapterBlock(32, sn=sn),
            AdapterBlock(64, sn=sn),
            AdapterBlock(128, sn=sn),
            AdapterBlock(256, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn)
        ])
        self.final_layer = mk_conv2d(400, 1, 2, sn=sn)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        self.pose_layer = mk_linear(2, 400, sn=sn)

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](
                    F.interpolate(input, scale_factor=0.5, mode='nearest')
                )

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x, None, None


class ResidualCCBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, sn=False):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p, sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p, sn=sn),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)
        self.proj = mk_conv2d(inplanes, planes, 1, stride=2, sn=sn)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity)/math.sqrt(2)
        return y


class CCSDiscriminator(nn.Module):
    def __init__(self, sn=False, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([
            ResidualCCBlock(32, 64, sn=sn), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128, sn=sn), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256, sn=sn), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400, sn=sn), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400, sn=sn), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400, sn=sn), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400, sn=sn), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList([
            AdapterBlock(32, sn=sn),
            AdapterBlock(64, sn=sn),
            AdapterBlock(128, sn=sn),
            AdapterBlock(256, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn)
        ])
        self.final_layer = mk_conv2d(400, 1, 2, sn=sn)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        self.pose_layer = mk_linear(2, 400, sn=sn)

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](
                    F.interpolate(input, scale_factor=0.5, mode='nearest')
                )

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x, None, None


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, in_channels, out_channels, with_r=False, sn=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = mk_conv2d(in_size, out_channels, sn=sn, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class ResidualCCBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, sn=False):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p, sn=sn),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p, sn=sn),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)
        self.proj = mk_conv2d(inplanes, planes, 1, stride=2, sn=sn)

    def forward(self, input):
        y = self.network(input)
        identity = self.proj(input)
        y = (y + identity)/math.sqrt(2)
        return y


class CCSDiscriminator(nn.Module):
    def __init__(self, sn=False, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([
            ResidualCCBlock(32, 64, sn=sn), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128, sn=sn), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256, sn=sn), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400, sn=sn), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400, sn=sn), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400, sn=sn), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400, sn=sn), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList([
            AdapterBlock(32, sn=sn),
            AdapterBlock(64, sn=sn),
            AdapterBlock(128, sn=sn),
            AdapterBlock(256, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn)
        ])
        self.final_layer = mk_conv2d(400, 1, 2, sn=sn)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        self.pose_layer = mk_linear(2, 400, sn=sn)

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](
                    F.interpolate(input, scale_factor=0.5, mode='nearest')
                )

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x, None, None


class CCSEncoderDiscriminator(nn.Module):
    def __init__(self, sn=False, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList([
            ResidualCCBlock(32, 64, sn=sn), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128, sn=sn), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256, sn=sn), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400, sn=sn), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400, sn=sn), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400, sn=sn), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400, sn=sn), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList([
            AdapterBlock(32, sn=sn),
            AdapterBlock(64, sn=sn),
            AdapterBlock(128, sn=sn),
            AdapterBlock(256, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn)
        ])
        # self.final_layer = mk_conv2d(400, 1 + 256 + 2, 2, sn=sn)
        self.final_layer = mk_conv2d(400, 1 + 2, 2, sn=sn)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}

    def forward(self, input, alpha, options=None, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](
                    F.interpolate(input, scale_factor=0.5, mode='nearest')
                )

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        prediction = x[..., 0:1]
        position = x[..., 1:3]

        return prediction, position

class CCSEncoder(nn.Module):
    def __init__(self, z_dim, sn=False, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()

        self.z_dim = z_dim
        self.layers = nn.ModuleList([
            ResidualCCBlock(32, 64, sn=sn), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128, sn=sn), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256, sn=sn), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400, sn=sn), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400, sn=sn), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400, sn=sn), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400, sn=sn), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList([
            AdapterBlock(32, sn=sn),
            AdapterBlock(64, sn=sn),
            AdapterBlock(128, sn=sn),
            AdapterBlock(256, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn)
        ])
        self.final_layer = mk_conv2d(400, self.z_dim + 2, 2, sn=sn)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}
        self.tanh = nn.Tanh()

    def forward(self, input, alpha):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](
                    F.interpolate(input, scale_factor=0.5, mode='nearest')
                )

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        latent = x[..., 0:self.z_dim]
        position = x[..., self.z_dim:self.z_dim+2]

        latent = self.tanh(latent)
        # latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
        # latent = 2 * latent - 1
        return latent, position

class CCSVAE(nn.Module):
    def __init__(self, z_dim, sn=False, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()

        self.z_dim = z_dim
        self.layers = nn.ModuleList([
            ResidualCCBlock(32, 64, sn=sn), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128, sn=sn), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256, sn=sn), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400, sn=sn), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400, sn=sn), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400, sn=sn), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400, sn=sn), # 7 4x4 -> 2x2
        ])

        self.fromRGB = nn.ModuleList([
            AdapterBlock(32, sn=sn),
            AdapterBlock(64, sn=sn),
            AdapterBlock(128, sn=sn),
            AdapterBlock(256, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn),
            AdapterBlock(400, sn=sn)
        ])
        self.final_layer = mk_conv2d(400, self.z_dim*2 + 2, 2, sn=sn)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        # sampling as if coming from the input space
        return mu + (eps * std)

    # def forward(self, input, alpha, options=None, **kwargs):
    def forward(self, input, alpha):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](
                    F.interpolate(input, scale_factor=0.5, mode='nearest')
                )

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        mu = x[..., 0:self.z_dim]
        log_var = x[..., self.z_dim:self.z_dim*2]
        latent = self.reparameterize(mu, log_var)
        position = x[..., self.z_dim*2:self.z_dim*2+2]
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        return latent, kld_loss, position

"""
Implements image encoders
"""
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

import torchvision

class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, z_dim=512):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        # self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.z_dim = z_dim
        self.fc = nn.Linear(512, z_dim+2)
        self.tanh = nn.Tanh()

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x, alpha):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, z_dim)
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        # x = self.tanh(x)

        return self.tanh(x[:, :self.z_dim]), x[:, self.z_dim:self.z_dim+2]