

from torch import nn, Tensor, Size
# import torch.nn as nn
from typing import Optional, Union, List

from . import register_norm_fn


@register_norm_fn(name="layer_norm")
class LayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape: Union[int, List[int], Size],
                 eps: Optional[float] = 1e-5,
                 elementwise_affine: Optional[bool] = True
                 ):
        super(LayerNorm, self).__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )

