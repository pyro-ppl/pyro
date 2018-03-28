from __future__ import absolute_import, division, print_function

from collections import namedtuple

import pytest
import torch

from pyro.contrib.gp.kernels import Constant, WhiteNoise
from pyro.contrib.gp.util import conditional


X = torch.tensor([[1, 5, 3], [2, 1, 4], [3, 2, 6]])

