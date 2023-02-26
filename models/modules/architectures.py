import torch.nn as nn
import torch.nn.functional as F
import torch
from ..layers.conv_layer import ConvLayer
from .cabm import RRG, RCB, RFA,  CBAMBlock, SKFF, SpatialAttention, CCALayer, ESA,\
    CCALayer, CBAMBlock,ChannelAttention, SpatialAttention,SKFF2,SKFF3,SKFF4,\
    SKFF5,SKFF6,SKFF7,SKFF8,SKFF9,SKFF10,SKFF11,iAFF,SKFF12, SKFF13,\
    SKFF14,SKFF15,SKFF16,SKFF17,SKFF18,SKFF19,SKFF20,SKFF21,SKFF22,\
    SKFF23,SKFF24,SKFF25, SKFF26, SKFF27, SKFF28, SKFF29

class Cond_block(nn.Module):
    def __init__(self, in_channels=3, out_channels=32, conv_ksize=3, stride=1, dilation=1, use_norm=False, use_act=True):
        super(Cond_block, self).__init__()
        self.use_norm = use_norm
        self.use_act = use_act
        kernel_size = (conv_ksize, conv_ksize)
        stride = (stride, stride)
        dilation = (dilation, dilation)
        padding = (int((kernel_size[0] - 1) / 2) * dilation[0], int((kernel_size[1] - 1) / 2) * dilation[1])
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, dilation=dilation, padding=padding)
        if self.use_act:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_act:
            x = self.act(x)
        return x
class new_SFT(nn.Module):
    def __init__(self,in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(new_SFT, self).__init__()

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

class res_SFT(nn.Module):
    def __init__(self,in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(res_SFT, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.new_sft0 = new_SFT(in_channels=in_channels, mid_channels=mid_channels, conv_ksize=3, act_type='leaky_relu',
                               norm_type='layer_norm', use_act=False, use_norm=False)
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.new_sft1 = new_SFT(in_channels=in_channels, mid_channels=mid_channels, conv_ksize=3, act_type='leaky_relu',
                                norm_type='layer_norm', use_act=False, use_norm=False)
        self.conv2 = ConvLayer(in_channels=mid_channels, out_channels=in_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)
    def forward(self, x):


        upx = x[1]
        fea = self.conv0(x[0])
        res = fea
        fea = self.new_sft0((fea, upx))
        fea = self.conv1(fea)

        fea = self.new_sft1((fea, upx))
        fea = fea + res
        fea = self.conv2(fea)
        return fea

class DEM(nn.Module):
    def __init__(self,in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(DEM, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv2 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv3 = ConvLayer(in_channels=mid_channels, out_channels=in_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=use_act, use_norm=False)
    def forward(self, x):

        realx = x[0]
        upx = x[1]
        res = realx
        x = self.conv0(realx)
        alpha = self.conv1(upx)
        beta = self.conv2(upx)
        x = torch.mul(x, alpha) + beta
        x = self.conv3(x) + res
        return x
class Res_SA(nn.Module):
    def __init__(self, in_channels=3, conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'):
        super(Res_SA, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.sa = SpatialAttention(kernel_size=7)
    def forward(self, x):
        res = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.sa(x) + res
        return x
class SFT(nn.Module):
    def __init__(self,in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(SFT, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv2 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.res_sa = Res_SA(in_channels=mid_channels, act_type=act_type, norm_type=norm_type)
        # self.cca = CCALayer(channel=mid_channels)
        self.conv3 = ConvLayer(in_channels=mid_channels, out_channels=in_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=use_act, use_norm=False)

    def forward(self, x):

        realx = x[0]
        upx = x[1]
        res = realx
        x = self.conv0(realx)
        alpha = self.conv1(upx)
        beta = self.conv2(upx)
        x = torch.mul(x, alpha) + beta
        x = self.res_sa(x)
        # x = self.cca(x)
        x = self.conv3(x) + res
        return x
class ResBlock_DEM(nn.Module):
    def __init__(self,in_channels=3, mid_channels=32, conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'):
        super(ResBlock_DEM, self).__init__()
        self.DEM0 = DEM(in_channels=in_channels, mid_channels=mid_channels, conv_ksize=conv_ksize,
                        act_type=act_type, norm_type=norm_type, use_act=True)
        # self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
        #                        kernel_size=conv_ksize, stride=1, act_type=act_type,
        #                        norm_type=norm_type, use_act=True, use_norm=False)
        self.DEM1 = DEM(in_channels=in_channels, mid_channels=mid_channels, conv_ksize=conv_ksize,
                        act_type=act_type, norm_type=norm_type)
        # self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=in_channels,
        #                        kernel_size=conv_ksize, stride=1, act_type=act_type,
        #                        norm_type=norm_type, use_act=False, use_norm=False)
    def forward(self, x):

        res = x[0]
        fea0 = self.DEM0(x)
        # fea0 = self.conv0(fea0)
        fea1 = self.DEM1((fea0, x[1]))
        # fea1 = self.con1(fea1)
        fea1 = fea1+res
        return fea1

class CGM(nn.Module):
    def __init__(self, in_channels=9, out_channels=64):
        super(CGM, self).__init__()
        # model = [nn.Conv2d(in_channels=9, out_channels=out_channel, kernel_size=3, padding=1),
        #          nn.LeakyReLU(0.1, inplace=True),
        #          SpatialAttention(7),
        #          nn.Conv2d(in_channels=out_channel, out_channels=3, kernel_size=3, padding=1),
        #          nn.LeakyReLU(0.1, inplace=True),
        #          ]
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        self.relu0 = nn.LeakyReLU(0.1, inplace=True),
        self.esa = ESA(n_feats=out_channels)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=3, kernel_size=3, padding=1),
        self.relu1 = nn.LeakyReLU(0.1, inplace=True),
    def forward(self, x):

        x = self.conv0(x)
        x = self.relu0(x)
        x = self.esa(x)
        x = self.conv1(x)
        x = self.relu1(x)
        return x

class new_SFT_v2(nn.Module):
    def __init__(self):
        super(new_SFT_v2, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 32, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift
class res_SFT_v2(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v2, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=3, stride=1)
        self.conv_up = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.new_sft0 = new_SFT_v2()
        self.conv0 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.new_sft1 = new_SFT_v2()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.rfa = RFA(out_channels=mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):

        realx_in = self.conv_real(x[0])
        upx_in = self.conv_up(x[1])
        fea = self.new_sft0((realx_in, upx_in))
        fea = self.conv0(fea)
        fea = self.new_sft1((fea, upx_in))
        fea = self.conv1(fea)
        fea = fea + realx_in
        fea = self.rfa(fea)
        fea = self.conv2(fea)
        return fea

class res_SFT_v3(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(res_SFT_v3, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv2 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.rfa = RFA(out_channels=mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx = x[0]
        upx = x[1]

        res = self.conv0(realx)
        alpha = self.conv1(upx)
        beta = self.conv2(upx)
        x = torch.mul(res, alpha) + beta + res
        x = self.rfa(x)
        x = self.conv3(x)
        return x

class res_SFT_v4(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v4, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=3, stride=1)
        self.conv_up = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.new_sft0 = new_SFT_v2()
        self.rfa = RFA(out_channels=mid_channels)
        self.conv0 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):

        realx_in = self.conv_real(x[0])
        upx_in = self.conv_up(x[1])
        fea = self.new_sft0((realx_in, upx_in))
        fea = self.rfa(fea)
        fea = self.conv0(fea)
        return fea
class new_sft_v5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(new_sft_v5, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(mid_channels, mid_channels, 1)
        self.SFT_shift_conv0 = nn.Conv2d(mid_channels, mid_channels, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv0(x[1])
        shift = self.SFT_shift_conv0(x[1])
        return x[0] * scale + shift
class Res_Block(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(Res_Block, self).__init__()
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
class res_SFT_v5(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v5, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                                   kernel_size=3, stride=1)
        self.conv_up = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.new_sft0 = new_sft_v5()
        self.res_block0 = Res_Block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v5()
        self.res_block1 = Res_Block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        upx_in = self.conv_up(x[1])

        fea = self.new_sft0((realx_in, upx_in))
        fea = self.res_block0(fea)
        fea = self.new_sft1((fea, upx_in))
        fea = self.res_block1(fea)
        fea = self.conv1(fea)
        return fea+x[0]

class res_SFT_v6(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v6, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                                   kernel_size=3, stride=1)
        self.conv_up = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.new_sft0 = new_sft_v5()
        res_block0 = []
        for _ in range(2):
            res_block0 += [Res_Block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')]
        self.res_block0 = nn.Sequential(*res_block0)
        self.new_sft1 = new_sft_v5()
        res_block1 = []
        for _ in range(2):
            res_block1 += [Res_Block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                     conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')]
        self.res_block1 = nn.Sequential(*res_block1)

        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        upx_in = self.conv_up(x[1])

        fea = self.new_sft0((realx_in, upx_in))
        fea = self.res_block0(fea)

        fea = self.new_sft0((fea, upx_in))
        fea = self.res_block0(fea)
        fea = self.conv1(fea)
        return fea+x[0]

class Res_CBAM(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
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
class res_SFT_v7(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v7, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                                   kernel_size=3, stride=1)
        self.conv_up = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.new_sft0 = new_sft_v5()
        self.res_cbam0 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v5()
        self.res_cbam1 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        upx_in = self.conv_up(x[1])

        fea = self.new_sft0((realx_in, upx_in))
        fea = self.res_cbam0(fea)
        fea = self.new_sft1((fea, upx_in))
        fea = self.res_cbam1(fea)
        fea = self.conv1(fea)
        return fea+x[0]

class res_SFT_v8(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v8, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                                   kernel_size=3, stride=1)
        self.conv_up = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.new_sft0 = new_sft_v5()
        self.res_cca0 = Res_CCA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v5()
        self.res_cca1 = Res_CCA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        upx_in = self.conv_up(x[1])
        fea = self.new_sft0((realx_in, upx_in))
        fea = self.res_cca0(fea)
        fea = self.new_sft1((fea, upx_in))
        fea = self.res_cca1(fea)
        fea = self.conv1(fea)
        return fea+x[0]
class new_sft_v9(nn.Module):
    def __init__(self,in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(new_sft_v9, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)      # 默认 use_act=Ture

        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)       # 默认 use_act=Ture

    def forward(self, x):

        alpha = self.conv0(x[1])
        beta = self.conv1(x[1])
        sft = torch.mul(x[0], alpha) + beta
        return sft
class Res_CA(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(Res_CA, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)     # default norm =False
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)
        self.CA = ChannelAttention(in_planes=mid_channels)
    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.CA(res)
        res = res + x
        return res
class res_SFT_v9(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v9, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)   # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        # self.res_ca0 = Res_CA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
        #                           conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        #
        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        # self.res_ca1 = Res_CA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
        #                           conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_ca0(fea)
        fea = self.new_sft1((fea, x[1]))
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class Res_block(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(Res_block, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)  # default norm =False

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = res + x
        return res
class res_SFT_v10(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v10, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=True)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_block0 = Res_block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_block1 = Res_block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_block0(fea)
        fea = self.new_sft1((fea, x[1]))
        fea = self.res_block1(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class Res_SA(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(Res_SA, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)  # default norm =False
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)
        self.sa = SpatialAttention(kernel_size=3)

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.sa(res) + x
        return res
class res_SFT_v11(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v11, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=True)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_sa0 = Res_SA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_sa1 = Res_SA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_sa0(fea)
        fea = self.new_sft1((fea, x[1]))
        fea = self.res_sa1(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class res_SFT_v12(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v12, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_block0 = Res_block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_block1 = Res_block(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_block0(fea)
        fea = self.new_sft1((fea, x[1]))
        fea = self.res_block1(fea)
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class res_SFT_v13(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v13, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=True)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_sa0 = Res_SA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_sa1 = Res_SA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_sa0(fea)
        fea = self.new_sft1((fea, x[1]))
        fea = self.res_sa1(fea)
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class res_SFT_v14(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v14, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.IN0 = nn.InstanceNorm2d(num_features=mid_channels, momentum=0.1)
        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_sa0 = Res_SA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.IN1 = nn.InstanceNorm2d(num_features=mid_channels, momentum=0.1)

        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_sa1 = Res_SA(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        realx_in = self.IN0(realx_in)
        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_sa0(fea)
        fea = self.IN1(fea)
        fea = self.new_sft1((fea, x[1]))
        fea = self.res_sa1(fea)
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class Res_CAlayer(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(Res_CAlayer, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)  # default norm =False
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)

        self.ca = CALayer(channel=mid_channels)

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.ca(res) + x

        return res
# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):  # 默认 reduction=8
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        mid_channel = max(int(channel / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, mid_channel, 1, padding=0),   # 默认bias=Ture
                nn.LeakyReLU(negative_slope=0.1, inplace=True),             # 默认为0.2
                nn.Conv2d(mid_channel, channel, 1, padding=0),   # 默认bias=Ture
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class Res_ECAlayer(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(Res_ECAlayer, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)  # default norm =False
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)

        self.ca = eca_layer()

    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.ca(res) + x
        return res
class Res_CCAlayer(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(Res_CCAlayer, self).__init__()
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)
        self.CCA = CCALayer(channel=mid_channels)
    def forward(self, x):
        res = self.conv0(x)
        res = self.conv1(res)
        res = self.CCA(res)
        res = res + x
        return res
class res_SFT_v15(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v15, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.IN0 = nn.InstanceNorm2d(num_features=mid_channels, momentum=0.1)
        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.IN1 = nn.InstanceNorm2d(num_features=mid_channels, momentum=0.1)
        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])

        realx_in = self.IN0(realx_in)
        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_ca0(fea)

        fea = self.IN1(fea)
        fea = self.new_sft1((fea, x[1]))
        fea = self.res_ca1(fea)

        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class res_SFT_v16(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v16, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,  # 卷积核默认为3
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)   # 卷积核默认为3


        self.res_cabm0 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                    conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)   # 卷积核默认为3

        self.res_cabm1 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
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
class res_SFT_v17(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v17, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv_1x1 = ConvLayer(in_channels=mid_channels*3, out_channels=mid_channels, kernel_size=1, stride=1,
                                    act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        fea_cat0 = realx_in

        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_ca0(fea)
        fea_cat1 = fea

        fea = self.new_sft1((fea, x[1]))
        fea = self.res_ca1(fea)
        fea_cat1 = torch.cat([fea_cat0, fea_cat1, fea], dim=1)

        fea_cat1 = self.conv_1x1(fea_cat1)
        fea = self.conv0(fea_cat1)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class res_SFT_v18(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v18, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.new_sft0((realx_in, x[1]))
        fea = self.res_ca0(fea)
        fea = self.conv0(fea)

        fea = self.new_sft1((fea, x[1]))
        fea = self.res_ca1(fea)
        fea = self.conv1(fea)

        upx = fea
        fea = self.conv2(fea)
        return (fea + x[0], upx)
class res_sft_block(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm'):
        super(res_sft_block, self).__init__()
        self.alpha = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.beta = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.res_ca = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

    def forward(self, x):
        alpha = self.alpha(x[1])
        beta = self.beta(x[1])
        sft = torch.mul(x[0], alpha) + beta
        res_sft = self.res_ca(sft)
        return res_sft
class res_SFT_v19(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v19, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.IN0 = nn.InstanceNorm2d(num_features=mid_channels, momentum=0.1, affine=False)
        self.sft0 = res_sft_block(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3,
                                  act_type='leaky_relu', norm_type='layer_norm')

        self.IN1 = nn.InstanceNorm2d(num_features=mid_channels, momentum=0.1, affine=False)
        self.sft1 = res_sft_block(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3,
                                  act_type='leaky_relu', norm_type='layer_norm')

        self.conv_1x1 = ConvLayer(in_channels=mid_channels*3, out_channels=mid_channels, kernel_size=1, stride=1,
                                    act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])
        fea_cat0 = realx_in

        realx_in = self.IN0(realx_in)
        fea = self.sft0((realx_in, x[1]))
        fea_cat1 = fea

        fea = self.IN1(fea)
        fea = self.sft1((fea, x[1]))

        fea_cat1 = torch.cat([fea_cat0, fea_cat1, fea], dim=1)

        fea_cat1 = self.conv_1x1(fea_cat1)
        fea = self.conv0(fea_cat1)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)
class res_SFT_v20(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v20, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)

        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)
        fea = self.conv0(fea)

        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v21(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v21, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_cbam0 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_cbam1 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_cbam0(fea)

        fea = self.skff1([fea, x[1]])
        fea = self.res_cbam1(fea)
        fea = self.conv0(fea)

        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)
class res_SFT_v22(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v22, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.res_cbam0 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff0 = SKFF(in_channels=mid_channels, height=2)

        self.res_cbam1 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.res_cbam0(realx_in)
        fea = self.skff0([fea, x[1]])

        fea = self.res_cbam1(fea)
        fea = self.skff1([fea, x[1]])
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v23(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v23, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff0 = SKFF(in_channels=mid_channels, height=2)

        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.res_ca3 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.res_ca0(realx_in)
        up1 = self.res_ca1(x[1])
        fea = self.skff0([fea, up1])

        fea = self.res_ca2(fea)
        up2 = self.res_ca3(up1)
        fea = self.skff1([fea, up2])

        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v24(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v24, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False


        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_cbam0 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.res_cbam1 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_cbam2 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.res_cbam3 = Res_CBAM(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.res_cbam1(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_cbam0(fea)

        up2 = self.res_cbam3(up1)
        fea = self.skff1([fea, up2])
        fea = self.res_cbam2(fea)

        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)
        return (fea + x[0], upx)

class res_SFT_v25(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v25, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.res_ca3 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.res_ca1(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)

        up2 = self.res_ca3(up1)
        fea = self.skff1([fea, up2])
        fea = self.res_ca2(fea)

        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v26(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v26, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        res_ca = []
        for _ in range(4):
            res_ca.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
        self.res_ca = nn.Sequential(*res_ca)

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca(fea)
        fea = self.conv0(fea)

        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v27(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v27, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff0 = SKFF(in_channels=mid_channels, height=2)

        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.res_ca3 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca4 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.res_ca0(realx_in)
        up1 = self.res_ca1(x[1])
        fea = self.skff0([fea, up1])

        fea = self.res_ca2(fea)
        up2 = self.res_ca3(up1)
        fea = self.skff1([fea, up2])
        fea = self.res_ca4(fea)
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v28(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v28, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.res_ca3 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.res_ca1(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)

        up2 = self.res_ca3(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca2(fea)

        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)
class res_SFT_v29(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v29, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff0 = SKFF(in_channels=mid_channels, height=2)

        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.res_ca0(realx_in)
        fea = self.skff0([fea, x[1]])

        fea = self.res_ca1(fea)
        fea = self.skff1([fea, x[1]])
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v30(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v30, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.res_ca0(realx_in)
        fea = self.skff0([fea, x[1]])
        fea = self.res_ca1(fea)
        fea = self.skff1([fea, x[1]])
        fea = self.res_ca2(fea)
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v31(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v31, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)

        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)

        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)

class res_SFT_v32(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v32, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)

        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)
        fea = self.conv2(fea)
        upx = fea
        fea = self.conv3(fea)

        return (fea + x[0], upx)

class res_SFT_v33(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v33, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)

        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv0(fea)

        return (fea + x[0], upx)

class res_SFT_v34(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v34, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.res_ca0(realx_in)
        up0 = self.conv0(x[1])
        fea = self.skff0([fea, up0])
        fea = self.res_ca1(fea)
        up1 = self.conv1(up0)
        fea = self.skff1([fea, up1])

        fea = self.res_ca2(fea)
        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)

class res_SFT_v35(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v35, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff0 = SKFF(in_channels=mid_channels, height=2)

        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.res_ca3 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca4 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.res_ca0(realx_in)
        up1 = self.res_ca1(x[1])
        fea = self.skff0([fea, up1])

        fea = self.res_ca2(fea)
        up2 = self.res_ca3(up1)
        fea = self.skff1([fea, up2])
        fea = self.res_ca4(fea)
        upx = fea
        fea = self.conv0(fea)

        return (fea + x[0], upx)

class res_SFT_v36(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v36, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff2 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca2 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)

        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)

        fea = self.skff2([fea, x[1]])
        fea = self.res_ca2(fea)
        upx = fea
        fea = self.conv0(fea)

        return (fea + x[0], upx)

class MFSB(nn.Module):
    def __init__(self, in_channels=32, ratio=4, mid_channels=48):
        super(MFSB, self).__init__()
        self.mid_channels = mid_channels
        self.ratio = ratio
        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
        self.convx1 = Res_CAlayer(in_channels=mid_channels//ratio, mid_channels=mid_channels//ratio, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff0 = SKFF(in_channels=mid_channels//ratio, height=2)
        self.convx2 = Res_CAlayer(in_channels=mid_channels//ratio, mid_channels=mid_channels//ratio, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels//ratio, height=2)
        self.convx3 = Res_CAlayer(in_channels=mid_channels//ratio, mid_channels=mid_channels//ratio, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=in_channels, kernel_size=1, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
    def forward(self, x):
        channel_size1 = self.mid_channels // self.ratio
        channel_size2 = 2*channel_size1
        channel_size3 = 3*channel_size1
        res = x
        x = self.conv0(x)
        x1 = x[:, 0:channel_size1, :, :]
        x2 = x[:, channel_size1:channel_size2, :, :]
        x3 = x[:, channel_size2:channel_size3, :, :]
        x4 = x[:, channel_size3:, :, :]
        y1 = self.convx1(x1)
        y2 = self.convx2(self.skff0([y1, x2]))
        y3 = self.convx3(self.skff1([y2, x3]))
        y = torch.cat([y1, y2, y3, x4], dim=1)
        y = self.conv1(y) + res
        return y

class res_SFT_v37(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v37, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)

        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)
        fea = self.conv2(fea)
        upx = fea
        fea = self.conv3(fea)

        return (fea + x[0], upx)

class res_SFT_v38(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v38, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF2(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF2(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)

        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)

        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)
class res_SFT_v39(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v39, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF3(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF3(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)
        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)

class res_SFT_v40(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v40, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF4(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF4(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)
        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)

class res_SFT_v41(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v41, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF3(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF3(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)

        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)
        fea = self.conv0(fea)

        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v42(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v42, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF5(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF5(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)
        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)

class res_SFT_v43(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v43, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF5(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF5(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)

        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)
        fea = self.conv0(fea)

        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v44(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v44, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF5(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff1 = SKFF5(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1x1 = ConvLayer(in_channels=mid_channels*3, out_channels=mid_channels, kernel_size=1, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)
        res1 = fea
        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)
        fea = torch.cat([realx_in, res1, fea], dim=1)
        fea = self.conv1x1(fea)
        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)

class res_SFT_v45(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v45, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF(in_channels=mid_channels, height=2)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1x1 = ConvLayer(in_channels=mid_channels * 3, out_channels=mid_channels, kernel_size=1, stride=1,
                                 act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)
        res1 = fea
        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)
        fea = torch.cat([realx_in, res1, fea], dim=1)
        fea = self.conv1x1(fea)
        fea = self.conv0(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v46(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v46, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF8(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF8(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv1x1 = ConvLayer(in_channels=mid_channels * 3, out_channels=mid_channels, kernel_size=1, stride=1,
                                 act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv0 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)
        res1 = fea
        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)
        fea = torch.cat([realx_in, res1, fea], dim=1)
        fea = self.conv1x1(fea)
        upx = fea
        fea = self.conv0(fea)

        return (fea + x[0], upx)

class res_SFT_v47(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v47, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False
        self.skff0 = SKFF8(in_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.skff1 = SKFF8(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv0 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.skff0([realx_in, x[1]])
        fea = self.res_ca0(fea)

        fea = self.skff1([fea, x[1]])
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv0(fea)

        return (fea + x[0], upx)

class res_SFT_v48(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v48, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff0 = SKFF21(in_channels=mid_channels)

        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.skff1 = SKFF21(in_channels=mid_channels)
        self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)
        up2 = self.conv1(x[1])
        fea = self.skff1([fea, up2])
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv2(fea)

        return (fea + x[0], upx)

class res_SFT_v49(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(res_SFT_v49, self).__init__()
        # self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
        #                            act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False
        self.conv_real = nn.Sequential(ConvLayer(in_channels=in_channels, out_channels=mid_channels // 2, kernel_size=3, stride=1,
                                act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
                      ConvLayer(in_channels=mid_channels // 2, out_channels=mid_channels, kernel_size=3, stride=1,
                                act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
                      ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
                      )
        # self.conv = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
        #                            act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.skff0 = SKFF21(in_channels=mid_channels)
        # self.skff0 = ENC(in_channels=mid_channels, out_channels=mid_channels)
        self.res_ca0 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        # self.skff1 = ENC(in_channels=mid_channels, out_channels=mid_channels)
        # self.res_ca1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
        #                            conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        # realx_in = self.conv(realx_in)
        up1 = self.conv0(x[1])

        fea = self.skff0([realx_in, up1])
        fea = self.res_ca0(fea)

        # fea = self.skff1([fea, up1])
        # fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v50(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v50, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.skff0 = SKFF21(in_channels=mid_channels)
        res_ca = []
        for _ in range(4):
            res_ca.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
        self.res_ca = nn.Sequential(*res_ca)
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])

        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class res_SFT_v51(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v51, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False
        res_ca0 = []
        for _ in range(3):
            res_ca0.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                      conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
        self.res_ca0 = nn.Sequential(*res_ca0)
        self.skff0 = SKFF21(in_channels=mid_channels)
        res_ca1 = []
        for _ in range(3):
            res_ca1.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                       conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
        self.res_ca1 = nn.Sequential(*res_ca1)
        self.conv0 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.conv1 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):
        realx_in = self.conv_real(x[0])
        realx_in = self.res_ca0(realx_in)
        up1 = self.conv0(x[1])
        fea = self.skff0([realx_in, up1])
        fea = self.res_ca1(fea)
        upx = fea
        fea = self.conv1(fea)

        return (fea + x[0], upx)

class FAM(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False):
        super(FAM, self).__init__()

        self.conv0 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)

        self.conv1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=True, use_norm=False)
        self.conv2 = ConvLayer(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=conv_ksize, stride=1, act_type=act_type,
                               norm_type=norm_type, use_act=False, use_norm=False)
    def forward(self, x):

        alpha = self.conv0(x[1])
        beta = self.conv1(x[1])
        alpha = torch.mul(x[0], alpha)
        fam = self.conv2(alpha) + beta
        return fam

class skff_block(nn.Module):
    def __init__(self, mid_channels=32):
        super(skff_block, self).__init__()
        self.conv = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.skff = SKFF21(in_channels=mid_channels)

        self.res_ca = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        # self.res_ca = Res_block(in_channels=mid_channels, mid_channels=mid_channels)
    def forward(self, x):

        up1 = self.conv(x[1])
        fea = self.skff([x[0], up1])
        # fea = self.skff([x[0], x[1]])
        fea = self.res_ca(fea)
        return fea, x[1]

class res_SFT_v52(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(res_SFT_v52, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        n_skff = 1
        n_res_ca = 0
        skff = []
        res_ca = []
        for _ in range(n_skff):
            skff.append(skff_block(mid_channels=mid_channels))
        self.skff = nn.Sequential(*skff)
        if n_res_ca > 0:
            for _ in range(n_res_ca):
                # res_ca.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                #                           conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
                res_ca.append(Res_block(in_channels=mid_channels, mid_channels=mid_channels))

            self.res_ca = nn.Sequential(*res_ca)
        else:
            self.res_ca = nn.Identity()

        self.conv = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):

        realx_in = self.conv_real(x[0])
        fea, _ = self.skff([realx_in, x[1]])
        upx = fea
        fea = self.res_ca(fea)
        fea = self.conv(fea) + x[0]

        return (fea , upx)

class skff_block2(nn.Module):
    def __init__(self, mid_channels=32):
        super(skff_block2, self).__init__()
        self.conv = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.res_ca = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                   conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')
        # self.res_ca = RCB(n_feat=mid_channels, groups=2)
        self.skff = SKFF21(in_channels=mid_channels)
    def forward(self, x):

        up1 = self.conv(x[1])
        fea = self.res_ca(x[0])
        fea = self.skff([fea, up1])

        return fea, x[1]

class res_SFT_v53(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(res_SFT_v53, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False
        self.skff_block0 = skff_block(mid_channels=mid_channels)

        self.skff_block1 = skff_block(mid_channels=mid_channels)
        n_res_ca = 0
        res_ca = []
        if n_res_ca > 0:
            for _ in range(n_res_ca):
                # res_ca.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                #                           conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
                res_ca.append(Res_block(in_channels=mid_channels, mid_channels=mid_channels))
            self.res_ca = nn.Sequential(*res_ca)
        else:
            self.res_ca = nn.Identity()

        self.conv = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):

        realx_in = self.conv_real(x[0])
        fea1, _ = self.skff_block0([realx_in, x[1]])
        fea2, _ = self.skff_block1([fea1, x[2]])
        fea2 = self.res_ca(fea2)
        fea = self.conv(fea2)

        return (fea + x[0], fea1, fea2)


class AFF(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                              act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
            ConvLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                  act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
        )
    def forward(self,x1, x2, x3):
        x = self.conv(torch.cat([x1, x2, x3], dim=1))
        return x

class RFDB(nn.Module):
    def __init__(self, in_channels):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        # self.esa = ESA(in_channels, nn.Conv2d)
        self.esa = CALayer(channel=in_channels)
    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

class ECP(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(ECP, self).__init__()
        self.conv_infeat = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                  act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.relu = nn.ReLU()
        res_block = []
        for _ in range(3):
            res_block += [Res_block(in_channels=out_channels, mid_channels=out_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm',)]
        self.res_block = nn.Sequential(*res_block)

        self.cdc = cdcconv(in_channels=out_channels, out_channels=out_channels)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.fuse = nn.Conv2d(3 * out_channels, out_channels, 1, 1, 0)

    def forward(self, x):

        x = self.conv_infeat(x)
        x_p = self.relu(x)
        x_res = self.res_block(x_p)
        x_cdc = self.cdc(x_p)
        x_norm = self.norm(x)
        x_out = self.fuse(torch.cat([x_res, x_cdc, x_norm], dim=1))

        return x_out

class res_SFT_v54(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(res_SFT_v54, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False
        self.skff_block1 = skff_block(mid_channels=mid_channels)

        n_res_ca = 2
        res_ca1 = []
        for _ in range(n_res_ca):
            res_ca1.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                      conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
        self.res_ca1 = nn.Sequential(*res_ca1)
        # self.fuse = ConvLayer(in_channels=2*mid_channels, out_channels=mid_channels, kernel_size=1, stride=1,
        #                            act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.skff_block2 = skff_block(mid_channels=mid_channels)
        res_ca2 = []
        for _ in range(n_res_ca):
            res_ca2.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                      conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
        self.res_ca2 = nn.Sequential(*res_ca2)
        self.conv = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):

        realx_in = self.conv_real(x[0])
        fea1, _ = self.skff_block1([realx_in, x[1]])
        fea1 = self.res_ca1(fea1)
        # fea2 = self.fuse(torch.cat([fea1, x[2]], dim=1))
        fea2, _ = self.skff_block1([fea1, x[2]])
        fea2 = self.res_ca2(fea2)
        fea = self.conv(fea2)

        return (fea + x[0], fea1, fea2)

class res_SFT_v55(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(res_SFT_v55, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False
        self.conv_up = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                              act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.skff = SKFF21(in_channels=mid_channels)
        n_res_ca = 4
        res_ca = []
        if n_res_ca > 0:
            for _ in range(n_res_ca):
                # res_ca.append(Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                #                           conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'))
                res_ca.append(Res_block(in_channels=mid_channels, mid_channels=mid_channels))
            self.res_ca = nn.Sequential(*res_ca)
        else:
            self.res_ca = nn.Identity()

        self.conv = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)
    def forward(self, x):

        realx_in = self.conv_real(x[0])
        upx = self.conv_up(x[1])
        fea1 = self.skff([realx_in, upx])
        fea2 = self.res_ca(fea1)
        fea = self.conv(fea2)

        return (fea + x[0], fea1)

class res_SFT_v56(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32):
        super(res_SFT_v56, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        # self.global_RMU = RMU(in_channels=32, mid_channels=32)
        self.local_RMU = RMU(in_channels=32, mid_channels=32)

        # self.conv = nn.Sequential(ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
        #                                     act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
        #                           ConvLayer(in_channels=mid_channels, out_channels=3, kernel_size=3, stride=1,
        #                                     act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
        #                           )
        #
        self.conv_up = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv = nn.Conv2d(mid_channels, 3, 1, 1)

    def forward(self, x):

        x_in = self.conv_real(x[0])
        # fea, _ = self.global_RMU([x_in, x[1]])
        # fea, _ = self.local_RMU([fea, x[2]])
        fea, _ = self.local_RMU([x_in, x[1]])
        # fea, _ = self.global_RMU([fea, x[2]])
        # fea = fea + x_in
        fea = self.conv_up(fea)
        up = fea
        fea = self.conv(fea)

        return fea, up

class RMU(nn.Module):
    def __init__(self, in_channels=128, mid_channels=32):
        super(RMU, self).__init__()
        self.FC1 = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.FC2 = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                             act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        # self.conv = nn.Sequential(ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
        #                            act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
        #                            ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
        #                                      act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
        #                            )
        self.conv = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                           conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

        self.FC3 = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                             act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.FC4 = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                             act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        # self.conv1 = nn.Sequential(ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
        #                            act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
        #                            ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
        #                                      act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
        #                            )
        self.conv1 = Res_CAlayer(in_channels=mid_channels, mid_channels=mid_channels, out_channels=3,
                                conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm')

    def forward(self, x):


        alpha1 = self.FC1(x[1])
        beta1 = self.FC2(x[1])
        fea = torch.mul(x[0], alpha1) + beta1
        fea = self.conv(fea)
        alpha2 = self.FC3(x[1])
        beta2 = self.FC4(x[1])
        fea = torch.mul(fea, alpha2) + beta2
        fea = self.conv1(fea)
        fea = fea + x[0]
        return fea, x[1]

class AggreBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(AggreBlock, self).__init__()
        self.conv0 = ConvLayer(in_channels=6, out_channels=out_channels, kernel_size=1, stride=1,
                             act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        model = []
        for _ in range(3):

            model += [Res_block(in_channels=out_channels, mid_channels=out_channels, out_channels=3,
                                  conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm',)]
        self.model = nn.Sequential(*model)

        self.conv1 = ConvLayer(in_channels=out_channels, out_channels=64, kernel_size=3, stride=2,
                             act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv2 = ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = ConvLayer(in_channels=128, out_channels=256, kernel_size=1, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
        self.conv4 = ConvLayer(in_channels=256, out_channels=128, kernel_size=1, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)


    def forward(self, x):

        x = self.conv0(x)
        x = self.model(x)
        x_map = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        x_vector = self.conv4(x)

        return x_vector, x_map

class res_SFT_v57(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=3):
        super(res_SFT_v57, self).__init__()
        self.conv_real = ConvLayer(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)  # default norm =False

        self.new_sft0 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)   # 默认
        self.conv0 = nn.Sequential(ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
                                   ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                             act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
                                   )

        self.new_sft1 = new_sft_v9(in_channels=mid_channels, mid_channels=mid_channels, conv_ksize=3)
        self.conv1 = nn.Sequential(ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                   act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
                                   ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                                             act_type='leaky_relu', norm_type='layer_norm', use_act=False, use_norm=False)
                                   )
        self.conv2 = ConvLayer(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=1,
                               act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)

        self.conv3 = nn.Conv2d(mid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        realx_in = self.conv_real(x[0])

        fea = self.new_sft0((realx_in, x[1]))
        fea = self.conv0(fea)

        fea = self.new_sft1((fea, x[1]))
        fea = self.conv1(fea)

        fea = self.conv2(fea)
        upx = fea

        fea = self.conv3(fea)
        return (fea + x[0], upx)

class ENC(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super(ENC, self).__init__()

        self.conv0 = nn.Sequential(
            ConvLayer(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1,
                      act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Sequential( ConvLayer(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1,
                      act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False),
                      ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                                act_type='leaky_relu', norm_type='layer_norm', use_act=True, use_norm=False)
                                    )
    def forward(self, x):
        out = torch.cat((x[0], x[1]), dim=1)

        xp1 = self.conv0(out)*x[1] + x[0]
        xp2 = (1-self.conv0(out)) * x[0] + x[1]

        xp = torch.cat((xp1, xp2), dim=1)
        xp = self.conv1(xp)

        return xp