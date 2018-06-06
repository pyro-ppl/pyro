import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import EmpiricalMarginal
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.trace_kernel import TraceKernel
from tests.common import assert_equal


class PriorKernel(TraceKernel):
    """
    Disregards the value of the current trace (or observed data) and
    samples a value from the model's prior.
    """
    def __init__(self, model):
        self.model = model
        self.data = None

    def setup(self, data):
        self.data = data

    def cleanup(self):
        self.data = None

    def initial_trace(self):
        return poutine.trace(self.model).get_trace(self.data)

    def sample(self, trace):
        return self.initial_trace()


def normal_normal_model(data):
    x = pyro.param('loc', torch.tensor([0.0]))
    y = pyro.sample('x', dist.Normal(x, torch.tensor([1.0])))
    pyro.sample('obs', dist.Normal(y, torch.tensor([1.0])), obs=data)
    return y


def test_mcmc_interface():
    data = torch.tensor([1.0])
    kernel = PriorKernel(normal_normal_model)
    mcmc = MCMC(kernel=kernel, num_samples=800, warmup_steps=100).run(data)
    marginal = EmpiricalMarginal(mcmc)
    assert_equal(marginal.sample_size, 800)
    sample_mean = marginal.mean
    sample_std = marginal.variance.sqrt()
    assert_equal(sample_mean, torch.tensor([0.0]), prec=0.08)
    assert_equal(sample_std, torch.tensor([1.0]), prec=0.08)
