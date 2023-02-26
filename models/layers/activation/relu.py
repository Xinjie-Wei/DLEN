

from torch import nn, Tensor
# import torch.nn as nn
from . import register_act_fn


@register_act_fn(name="relu")
class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__(inplace=inplace)


