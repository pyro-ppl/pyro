from __future__ import absolute_import, division, print_function

import torch


class TransformModule(torch.distributions.Transform, torch.nn.Module):
    def __init__(self):
        super(TransformModule, self).__init__()
        # torch.distributions.Transform.__init__(self)
        # torch.nn.Module.__init__(self)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()
