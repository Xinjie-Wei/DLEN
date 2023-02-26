

import torch
from torch import nn
import math

from models.layers.normalization import (
    BatchNorm1d, BatchNorm2d, SyncBatchNorm, LayerNorm, InstanceNorm1d, InstanceNorm2d, GroupNorm, SUPPORTED_NORM_FNS
)


norm_layers_tuple = (BatchNorm1d, BatchNorm2d, SyncBatchNorm, LayerNorm, InstanceNorm1d, InstanceNorm2d, GroupNorm)

def get_normalization_layer(num_features=64, norm_type=None, num_groups=None,):
    norm_type = 'layer_norm' if norm_type is None else norm_type
    num_groups = 32 if num_groups is None else num_groups
    momentum = 0.1

    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None
    if norm_type in ['batch_norm', 'batch_norm_2d']:
        norm_layer = BatchNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == 'batch_norm_1d':
        norm_layer = BatchNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ['sync_batch_norm', 'sbn']:
        norm_layer = SyncBatchNorm(num_features=num_features, momentum=momentum)
    elif norm_type in ['group_norm', 'gn']:
        num_groups = math.gcd(num_features, num_groups)
        norm_layer = GroupNorm(num_channels=num_features, num_groups=num_groups)
    elif norm_type in ['instance_norm', 'instance_norm_2d']:
        norm_layer = InstanceNorm2d(num_features=num_features, momentum=momentum)
    elif norm_type == "instance_norm_1d":
        norm_layer = InstanceNorm1d(num_features=num_features, momentum=momentum)
    elif norm_type in ['layer_norm', 'ln']:
        norm_layer = LayerNorm(num_features)
    else:
        print(
            'Supported normalization layer arguments are: {}. Got: {}'.format(SUPPORTED_NORM_FNS, norm_type))

    return norm_layer
