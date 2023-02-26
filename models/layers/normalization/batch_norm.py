

from torch import nn, Tensor
from typing import Optional

from . import register_norm_fn


@register_norm_fn(name="batch_norm")
@register_norm_fn(name="batch_norm_2d")
class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: Optional[float] = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: Optional[bool] = True,
                 track_running_stats: Optional[bool] = True
                 ):
        super(BatchNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                          track_running_stats=track_running_stats)

@register_norm_fn(name="batch_norm_1d")
class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self,
                 num_features: int,
                 eps: Optional[float] = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: Optional[bool] = True,
                 track_running_stats: Optional[bool] = True
                 ):
        super(BatchNorm1d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                          track_running_stats=track_running_stats)
