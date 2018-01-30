from __future__ import absolute_import, division, print_function

import torch.nn as nn
from torch.nn import Parameter


class InducingPoints(nn.Module):

    def __init__(self, Xu, name="inducing_points"):
        super(self, InducingPoints).__init__()
        self.inducing_points = Parameter(Xu)

    def forward(self):
        return self.inducing_points
