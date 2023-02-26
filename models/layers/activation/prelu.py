

from torch import nn, Tensor

from . import register_act_fn


@register_act_fn(name="prelu")
class PReLU(nn.PReLU):
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

