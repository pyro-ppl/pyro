import logging

import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import Marginal
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.trace_kernel import TraceKernel
from tests.common import assert_equal

logging.basicConfig()
# Change the logging level to DEBUG to see the output of the MCMC logger
logging.getLogger().setLevel(logging.ERROR)


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
    mcmc = MCMC(kernel=kernel, num_samples=800, warmup_steps=100)
    marginal = Marginal(mcmc)
    dist, values = marginal._dist_and_values(data)
    assert_equal(len(values), 800)
    samples = []
    for _ in range(600):
        samples.append(values[dist.sample().item()])
    sample_mean = torch.mean(torch.stack(samples), 0)
    sample_std = torch.std(torch.stack(samples), 0)
    assert_equal(sample_mean.data, torch.tensor([0.0]), prec=0.08)
    assert_equal(sample_std.data, torch.tensor([1.0]), prec=0.08)
