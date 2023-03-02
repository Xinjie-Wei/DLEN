import torch.nn as nn
import torch.nn.functional as F
import torch
from .modules.Trans_low import Trans_low
from .layers.conv_layer import ConvLayer

class Lap_Pyramid_Bicubic(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Bicubic, self).__init__()

        self.interpolate_mode = 'bicubic'
        self.num_high = num_high

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            down = nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode=self.interpolate_mode, align_corners=True)
            up = nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode, align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
        return image

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image

class Res_block(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm'):
        super(Res_block, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)

    def forward(self, x):

        res = self.conv0(x)
        res = self.conv1(res)
        res = res + x

        return res

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=32):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        d = max(int(in_planes / 8), 4)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_planes, d, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(d, in_planes, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

        self.conv = nn.Conv2d(2,
                              1,
                              kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avgout, maxout], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x

class CBAMBlock(nn.Module):
    def __init__(self, channel):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes=channel)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class Res_CBAM(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm'):
        super(Res_CBAM, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)

        self.CBAM = CBAMBlock(channel=mid_channels)

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.CBAM(res)
        res = res + x
        return res

class SFT(nn.Module):
    """
        SFT: Affine transformation
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        mid_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        conv_ksize: The kernel size of convolution. Default: 3

    """
    def __init__(self,in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm'):
        super(SFT, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

    def forward(self, x):

        alpha = self.conv0(x[1])
        beta = self.conv1(x[1])
        sft = torch.mul(x[0], alpha) + beta
        return sft

class CFEM(nn.Module):
    """
        CFEM : Cross-Layer Feature Enhancement Module
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        mid_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
    """
    def __init__(self, in_channels=3, mid_channels=32):
        super(CFEM, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.new_sft0 = SFT(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_cabm0 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels,
                                    conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.new_sft1 = SFT(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_cabm1 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):

        realx_in = self.conv_real(x[0])

        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_cabm0(fea)

        fea = self.new_sft1((fea, x[1]))
        fea = self.res_cabm1(fea)

        fea = self.conv0(fea)
        upx = fea

        fea = self.conv1(fea)

        return (fea + x[0], upx)


class Trans_high(nn.Module):
    def __init__(self,  num_high=3):
        super(Trans_high, self).__init__()
        self.num_high = num_high
        out_channels = 32
        mid_channels = 32
        model = [
                    nn.Conv2d(in_channels=9, out_channels=out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.1, inplace=True),
                ]
        for _ in range(3):
            model += [Res_block(in_channels=out_channels, mid_channels=out_channels,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm',)]

        self.model = nn.Sequential(*model)
        for i in range(1, self.num_high):
            upsample_block = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=mid_channels,
                                                kernel_size=4, padding=1, stride=2)
            setattr(self, 'upsample_block_{}'.format(str(i)), upsample_block)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(CFEM(in_channels=3, mid_channels=mid_channels))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = []
        mask = self.model(x)
        for i in range(self.num_high):

            if i != 0:
                self.upsample_block = getattr(self, 'upsample_block_{}'.format(str(i)))
                mask = self.upsample_block(mask)

            self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))

            (result_highfreq, up) = self.trans_mask_block((pyr_original[-2 - i], mask))
            mask = up
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):

            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)

        return pyr_result


class DLEN(nn.Module):
    def __init__(self, num_high=3):                   # num_high denotes the number of high frequency layers
        super(DLEN, self).__init__()

        # Lap_Pyramid_Conv: the function of Laplacian pyramid decomposition and reconstruction

        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        self.trans_low = Trans_low()

        self.trans_high = Trans_high(num_high=num_high)

        self.upsample = nn.ConvTranspose2d(in_channels=3, out_channels=3,
                                           kernel_size=4, padding=1, stride=2)

        self.upsample2 = nn.ConvTranspose2d(in_channels=3, out_channels=3,
                                            kernel_size=4, padding=1, stride=2)

    def forward(self, x):

        pyr_A = self.lap_pyramid.pyramid_decom(img=x)
        fake_B_low = self.trans_low(pyr_A[-1])
        real_A_up = self.upsample(pyr_A[-1])
        fake_B_up = self.upsample2(fake_B_low)
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)

        return fake_B_full

