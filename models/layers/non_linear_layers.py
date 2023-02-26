

from typing import Optional
import logging

from .activation import (
    ReLU, Hardswish, Hardsigmoid, PReLU, LeakyReLU, Swish, GELU, Sigmoid, ReLU6, SUPPORTED_ACT_FNS
)


def get_activation_fn(act_type: str = 'swish', num_parameters: Optional[int] = -1, inplace: Optional[bool] = True,
                      negative_slope: Optional[float] = 0.1):
    if act_type == 'relu':
        return ReLU(inplace=False)
    elif act_type == 'prelu':
        assert num_parameters >= 1
        return PReLU(num_parameters=num_parameters)
    elif act_type == 'leaky_relu':
        return LeakyReLU(negative_slope=negative_slope, inplace=inplace)
    elif act_type == 'hard_sigmoid':
        return Hardsigmoid(inplace=inplace)
    elif act_type == 'swish':
        return Swish(inplace=inplace)
    elif act_type == 'gelu':
        return GELU()
    elif act_type == 'sigmoid':
        return Sigmoid()
    elif act_type == 'relu6':
        return ReLU6(inplace=inplace)
    elif act_type == 'hard_swish':
        return Hardswish(inplace=inplace)
    else:
        logging.info(
            'Supported activation layers are: {}. Supplied argument is: {}'.format(SUPPORTED_ACT_FNS, act_type))
