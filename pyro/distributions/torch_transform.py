from __future__ import absolute_import, division, print_function

import torch


class TransformModule(torch.distributions.Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.

    """

    def __init__(self, *args, **kwargs):
        super(TransformModule, self).__init__(*args, **kwargs)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()
