# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import math


def xavier_uniform(D_in, D_out):
    scale = math.sqrt(6.0 / float(D_in + D_out))
    noise = torch.rand(D_in, D_out)
    return 2.0 * scale * noise - scale


def adjoin_ones_vector(x):
    return torch.cat([x, torch.ones(x.shape[:-1] + (1,)).type_as(x)], dim=-1)


def adjoin_zeros_vector(x):
    return torch.cat([x, torch.zeros(x.shape[:-1] + (1,)).type_as(x)], dim=-1)
