from __future__ import absolute_import, division, print_function

import os

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC, _UnarySampler, _MultiSampler
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.util import optional
from tests.common import skipif_param, assert_close


class PriorKernel(MCMCKernel):
    """
    Disregards the value of the current trace (or observed data) and
    samples a value from the model's prior.
    """
    def __init__(self, model):
        self.model = model
        self.data = None
        self._initial_params = None
        self._prototype_trace = None

    def setup(self, warmup_steps, data):
        self.data = data
        self._prototype_trace = poutine.trace(self.model).get_trace(data)

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def cleanup(self):
        self.data = None

    def sample_params(self):
        trace = poutine.trace(self.model).get_trace(self.data)
        return {k: v["value"] for k, v in trace.iter_stochastic_nodes()}

    def sample(self, trace):
        return self.sample_params()


def normal_normal_model(data):
    x = torch.tensor([0.0])
    y = pyro.sample('y', dist.Normal(x, torch.ones(data.shape)))
    pyro.sample('obs', dist.Normal(y, torch.tensor([1.0])), obs=data)
    return y


def test_mcmc_interface():
    data = torch.tensor([1.0])
    initial_params, _, transforms, _ = initialize_model(normal_normal_model, model_args=(data,))
    kernel = PriorKernel(normal_normal_model)
    samples = MCMC(kernel=kernel, num_samples=800, warmup_steps=100,
                   initial_params=initial_params, transforms=transforms).run(data)
    sample_mean = samples['y'].mean()
    sample_std = samples['y'].std()
    assert_close(sample_mean, torch.tensor(0.0), atol=0.05)
    assert_close(sample_std, torch.tensor(1.0), atol=0.05)


@pytest.mark.parametrize("num_chains, cpu_count", [
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
])
def test_num_chains(num_chains, cpu_count, monkeypatch):
    monkeypatch.setattr(torch.multiprocessing, 'cpu_count', lambda: cpu_count)
    data = torch.tensor([1.0])
    initial_params, _, transforms, _ = initialize_model(normal_normal_model,
                                                        model_args=(data,),
                                                        num_chains=num_chains)
    kernel = PriorKernel(normal_normal_model)
    available_cpu = max(1, cpu_count-1)
    with optional(pytest.warns(UserWarning), available_cpu < num_chains):
        mcmc = MCMC(kernel, num_samples=10, num_chains=num_chains,
                    initial_params=initial_params, transforms=transforms)
    assert mcmc.num_chains == min(num_chains, available_cpu)
    if mcmc.num_chains == 1:
        assert isinstance(mcmc.sampler, _UnarySampler)
    else:
        assert isinstance(mcmc.sampler, _MultiSampler)


def _empty_model():
    return torch.tensor(1)


@pytest.mark.parametrize("kernel, model", [
    (HMC, _empty_model),
    (NUTS, _empty_model),
])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("num_chains", [
    1,
    skipif_param(2, condition="CI" in os.environ or "CUDA_TEST" in os.environ,
                 reason="CI only provides 1 CPU; also see https://github.com/pytorch/pytorch/issues/2517")
])
def test_null_model_with_hook(kernel, model, jit, num_chains):
    num_warmup, num_samples = 10, 10
    initial_params, potential_fn, transforms, _ = initialize_model(model,
                                                                   num_chains=num_chains)

    iters = []

    def hook(kernel, samples, stage, i):
        assert samples == {}
        iters.append((stage, i))

    kern = kernel(potential_fn=potential_fn, transforms=transforms, jit_compile=jit)
    samples = MCMC(kern, num_samples=num_samples, warmup_steps=num_warmup,
                   num_chains=num_chains, initial_params=initial_params, hook_fn=hook).run()
    assert samples == {}
    if num_chains == 1:
        expected = [("warmup", i) for i in range(num_warmup)] + [("sample", i) for i in range(num_samples)]
        assert iters == expected
