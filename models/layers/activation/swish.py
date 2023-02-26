
from torch import nn, Tensor

from . import register_act_fn


@register_act_fn(name="swish")
class Swish(nn.SiLU):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__(inplace=inplace)


