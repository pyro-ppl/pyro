import torch

import pyro.distributions as dist
from tests.common import requires_cuda


@requires_cuda
def test_dirichlet_grad_cuda():
    concentration = torch.ones(3, requires_grad=True)
    dist.Dirichlet(concentration).rsample().sum().backward()


@requires_cuda
def test_linspace():
    x = torch.linspace(-1., 1., 100, device="cuda")
    assert x.device.type == "cuda"
