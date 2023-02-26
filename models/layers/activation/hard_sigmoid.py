

from torch import nn, Tensor

from . import register_act_fn


@register_act_fn(name="hard_sigmoid")
class Hardsigmoid(nn.Hardsigmoid):
    def __init__(self, inplace: bool = False):
        super(Hardsigmoid, self).__init__(inplace=inplace)


