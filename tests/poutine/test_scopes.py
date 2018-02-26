import pyro
import pyro.util
import pyro.poutine as poutine
import pyro.distributions.torch as dist

import torch
from torch.autograd import Variable

from pyro.poutine import scope_poutine
