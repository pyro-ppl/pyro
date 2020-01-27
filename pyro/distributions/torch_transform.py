# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


class TransformModule(torch.distributions.Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __hash__(self):
        return super().__hash__()
