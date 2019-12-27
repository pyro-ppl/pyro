import pytest
import torch

import pyro.distributions as dist
from pyro.infer.reparam import TransReparam


def test_log_normal(shape):
    loc = torch.empty(shape).uniform_(-1, 1)
    scale = torch.empty(shape).uniform_(0.5, 1.5)

    def model():
        with pyro.plate_stack("plates", shape):
            with pyro.plate("particles", 200000):
                if "dist_type" == "Normal":
                    pyro.sample("x", dist.Normal(loc, scale))
                else:
                    pyro.sample("x", dist.StudentT(10.0, loc, scale))
