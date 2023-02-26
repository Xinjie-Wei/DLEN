

from torch import nn, Tensor

from . import register_act_fn


@register_act_fn(name="sigmoid")
class Sigmoid(nn.Sigmoid):
    def __init__(self):
        super(Sigmoid, self).__init__()


