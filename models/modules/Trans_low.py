import torch
import torch.nn as nn
import os
from .mobilevit_block import LFEM
from ..layers.conv_layer import ConvLayer

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, stride=1, expand_ratio=6, activation=nn.ReLU6):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride in [1, 2]
        hidden_dim = int(in_channels * expand_ratio)
        self.is_residual = self.stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            # pw Point-wise
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            activation(inplace=True),
            # dw  Depth-wise
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            activation(inplace=True),
            # pw-linear, Point-wise linear
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),

        )
    def forward(self, x):
        if self.stride == 1 and self.in_channels == self.out_channels:
            res = self.conv(x)
            x = x + res
        else:
            x = self.conv(x)
        return x

class Trans_low(nn.Module):
    """
        Trans_low : The bottom layer of the Laplacian Pyramid processing network (including LFEM network)
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        mid_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        down_channel: MobileNetv2 Block output channel
        conv_ksize: The kernel size of convolution. Default: 3

    """
    def __init__(self, in_channel=3, out_channel=64, down_channel=96, conv_ksize=3,
                 act_type='leaky_relu', norm_type='layer_norm'):
        super(Trans_low, self).__init__()
        if out_channel == 48:
            ratio = 6
        elif out_channel == 64:
            ratio = 8
        out_channel_in0 = out_channel_out1 = int(2*out_channel/ratio)
        out_channel_in1 = out_channel_out0 = int(4*out_channel/ratio)

        self.conv_in_block0 = ConvLayer(in_channels=in_channel, out_channels=out_channel_in0,
                                        kernel_size=conv_ksize, stride=1, act_type=act_type,
                                        norm_type=norm_type, use_act=True, use_norm=False)
        self.conv_in_block1 = ConvLayer(in_channels=out_channel_in0, out_channels=out_channel_in1,
                                        kernel_size=conv_ksize, stride=1, act_type=act_type,
                                        norm_type=norm_type, use_act=True, use_norm=False)

        self.conv_in_block2 = ConvLayer(in_channels=out_channel_in1, out_channels=out_channel,
                                        kernel_size=conv_ksize, stride=1, act_type=act_type,
                                        norm_type=norm_type, use_act=True, use_norm=False)

        self.downsample = InvertedResidualBlock(in_channels=out_channel, out_channels=down_channel,
                                           stride=2, expand_ratio=6)

        self.upsample = nn.ConvTranspose2d(in_channels=down_channel, out_channels=out_channel,
                                           kernel_size=4, padding=1, stride=2)
        head_dim = None
        if down_channel == 64:
            transformer_dim = 96
            ffn_dim = 192
        elif down_channel == 96:
            transformer_dim = 144
            ffn_dim = 288
        if head_dim is None:
            num_heads = 8
            head_dim = transformer_dim // num_heads

        self.mit_block = LFEM(
                in_channels=down_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=3,
                head_dim=head_dim,
                dropout=0.0,
                ffn_dropout=0.0,
                attn_dropout=0.0,
                patch_h=2,
                patch_w=2,
                transformer_norm_layer='layer_norm',
                conv_ksize=conv_ksize,
                act_type=act_type,
                norm_type=norm_type,
                 )

        self.conv_out_block0 = ConvLayer(in_channels=out_channel, out_channels=out_channel_out0,
                                         kernel_size=conv_ksize, stride=1, act_type=act_type,
                                         norm_type=norm_type, use_act=True, use_norm=False)
        self.conv_out_block1 = ConvLayer(in_channels=out_channel_out0, out_channels=out_channel_out1,
                                         kernel_size=conv_ksize, stride=1, act_type=act_type,
                                         norm_type=norm_type, use_act=True, use_norm=False)
        self.conv_out_block2 = ConvLayer(in_channels=out_channel_out1, out_channels=in_channel,
                                         kernel_size=conv_ksize, stride=1, act_type=act_type,
                                         norm_type=norm_type, use_act=False, use_norm=False)

    def forward(self, x):

        res0 = x
        x = self.conv_in_block0(x)  # 3-16
        x = self.conv_in_block1(x)  # 16-32
        x = self.conv_in_block2(x)  # 32-64

        x = self.downsample(x)
        x = self.mit_block(x)
        x = self.upsample(x)

        x = self.conv_out_block0(x)  # 64-32
        x = self.conv_out_block1(x)  # 32-16
        x = self.conv_out_block2(x)  # 16-3

        x = x + res0

        return x


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    img = torch.Tensor(8, 3, 600, 400)
    global_net = Trans_low()
    x = global_net(img)
