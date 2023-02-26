

from torch import nn, Tensor

from . import register_act_fn


@register_act_fn(name="leaky_relu")
class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False):
        super(LeakyReLU, self).__init__(negative_slope=negative_slope, inplace=inplace)


