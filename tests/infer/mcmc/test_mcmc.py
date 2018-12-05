import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.mcmc.mcmc import MCMC, _SingleSampler, _ParallelSampler
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.util import optional
from tests.common import assert_equal


class PriorKernel(TraceKernel):
    """
    Disregards the value of the current trace (or observed data) and
    samples a value from the model's prior.
    """
    def __init__(self, model):
        self.model = model
        self.data = None

    def setup(self, warmup_steps, data):
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
    marginal = mcmc.marginal().empirical["_RETURN"]
    assert_equal(marginal.sample_size, 800)
    sample_mean = marginal.mean
    sample_std = marginal.variance.sqrt()
    assert_equal(sample_mean, torch.tensor([0.0]), prec=0.08)
    assert_equal(sample_std, torch.tensor([1.0]), prec=0.08)


@pytest.mark.parametrize("num_chains, cpu_count", [
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
])
def test_num_chains(num_chains, cpu_count, monkeypatch):
    monkeypatch.setattr(torch.multiprocessing, 'cpu_count', lambda:  cpu_count)
    kernel = PriorKernel(normal_normal_model)
    available_cpu = max(1, cpu_count-1)
    with optional(pytest.warns(UserWarning), available_cpu < num_chains):
        mcmc = MCMC(kernel, num_samples=10, num_chains=num_chains)
    assert mcmc.num_chains == min(num_chains, available_cpu)
    if mcmc.num_chains == 1:
        assert isinstance(mcmc.sampler, _SingleSampler)
    else:
        assert isinstance(mcmc.sampler, _ParallelSampler)
