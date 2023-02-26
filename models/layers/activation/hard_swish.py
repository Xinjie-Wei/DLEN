
from torch import nn, Tensor

from . import register_act_fn


@register_act_fn(name="hard_swish")
class Hardswish(nn.Hardswish):
    def __init__(self, inplace: bool = False):
        super(Hardswish, self).__init__(inplace=inplace)


