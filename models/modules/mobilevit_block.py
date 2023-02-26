
from torch import nn
import math
import torch
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from .transformer import TransformerEncoder
from ..layers import ConvLayer, get_normalization_layer

class LFEM(nn.Module):
    """
        LFEM: Low Frequency Enhancement Module
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)
        transformer_dim: Input dimension to the transformer unit. Default: 144
        ffn_dim: Dimension of the FFN block. Default: 288
        n_transformer_blocks: Number of transformer blocks. Default: 3
        head_dim: Head dimension in the multi-head attention. Default: 18
        attn_dropout: Dropout in multi-head attention. Default: 0.0
        dropout: Dropout rate. Default: 0.0
        ffn_dropout: Dropout between FFN layers in transformer. Default: 0.0
        atch_h: Patch height for unfolding operation. Default: 2
        patch_w: Patch width for unfolding operation. Default: 2
        transformer_norm_layer: Normalization layer in the transformer block. Default: layer_norm
        conv_ksize: The kernel size of convolution. Default: 3
        The partial network settings were inspired by the paper:
        MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer
            https://arxiv.org/abs/2110.02178

    """
    def __init__(self, in_channels=96, transformer_dim=144, ffn_dim=288,
                 n_transformer_blocks=3, head_dim=18, attn_dropout=0.0, dropout=0.0,
                 ffn_dropout=0.0, patch_h=2, patch_w=2, transformer_norm_layer="layer_norm",
                 conv_ksize=3, act_type='leaky_relu', norm_type='layer_norm'):
        super(LFEM, self).__init__()

        self.local_rep0 = ConvLayer(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, act_type=act_type,
            norm_type=norm_type, use_norm=False, use_act=True
        )

        self.local_rep1 = ConvLayer(
             in_channels=in_channels, out_channels=transformer_dim,
            kernel_size=1, stride=1, act_type=act_type,
            norm_type=norm_type, use_norm=False, use_act=False
        )

        self.conv_1x1_out = ConvLayer(
            in_channels=transformer_dim, out_channels=in_channels,
            kernel_size=1, stride=1, act_type=act_type,
            norm_type=norm_type, use_norm=False, use_act=True
        )

        self.conv_3x3_out = ConvLayer(
            in_channels=2*in_channels, out_channels=in_channels,
            kernel_size=conv_ksize, stride=1, act_type=act_type,
            norm_type=norm_type, use_norm=False, use_act=True
        )
        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(embed_dim=transformer_dim, ffn_latent_dim=ffn_dims[block_idx], num_heads=num_heads,
                               attn_dropout=attn_dropout, dropout=dropout, ffn_dropout=ffn_dropout, act_type=act_type,
                               transformer_norm_layer=transformer_norm_layer)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=transformer_dim)
        )
        self.global_rep = nn.Sequential(*global_rep)


        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(feature_map, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w # n_w
        num_patch_h = new_h // patch_h # n_h
        num_patches = num_patch_h * num_patch_w # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose(1, 3)
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.contiguous().view(info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1)

        batch_size, pixels, num_patches, channels = patches.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose(1, 3)

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            feature_map = F.interpolate(feature_map, size=info_dict["orig_size"], mode="bilinear", align_corners=False)
        return feature_map

    def forward(self, x):
        fm = self.local_rep0(x)
        res = x
        fm = self.local_rep1(fm)
        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_1x1_out(fm)

        fm = self.conv_3x3_out(
            torch.cat((res, fm), dim=1)
        )

        return fm

