
from torch import nn
from ..layers import get_normalization_layer, LinearLayer, get_activation_fn, MultiHeadAttention, Dropout

class TransformerEncoder(nn.Module):
    """
        This class defines the Transformer encoder (pre-norm) as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    """

    def __init__(self, embed_dim=144, ffn_latent_dim=288, num_heads=8, attn_dropout=0.0, dropout=0.0,
                 ffn_dropout=0.0, transformer_norm_layer="layer_norm", act_type='leaky_relu'):
        super(TransformerEncoder, self).__init__()
        self.act_type = act_type
        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            MultiHeadAttention(embed_dim, num_heads, attn_dropout=attn_dropout, bias=True),
            Dropout(p=dropout)
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(norm_type=transformer_norm_layer, num_features=embed_dim),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            self.build_act_layer(self.act_type),
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout)
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout

    @staticmethod
    def build_act_layer(act_type):
        act_type = act_type
        neg_slope = 0.1
        inplace = False
        act_layer = get_activation_fn(act_type=act_type, inplace=inplace, negative_slope=neg_slope,
                                      num_parameters=1)
        return act_layer

    def forward(self, x):

        # Multi-head attention
        x = x + self.pre_norm_mha(x)

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

