import os
from functools import partial

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

    def diagnostics(self):
        return {'dummy_key': 'dummy_value'}

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

    def sample(self, params):
        new_params = self.sample_params()
        assert params.keys() == new_params.keys()
        for k, v in params.items():
            assert new_params[k].shape == v.shape
        return new_params


def normal_normal_model(data):
    x = torch.tensor([0.0])
    y = pyro.sample('y', dist.Normal(x, torch.ones(data.shape)))
    pyro.sample('obs', dist.Normal(y, torch.tensor([1.0])), obs=data)
    return y


@pytest.mark.parametrize('num_draws', [None, 1800, 2200])
@pytest.mark.parametrize('group_by_chain', [False, True])
@pytest.mark.parametrize('num_chains', [1, 2])
def test_mcmc_interface(num_draws, group_by_chain, num_chains):
    num_samples = 2000
    data = torch.tensor([1.0])
    initial_params, _, transforms, _ = initialize_model(normal_normal_model, model_args=(data,))
    kernel = PriorKernel(normal_normal_model)
    mcmc = MCMC(kernel=kernel, num_samples=num_samples, warmup_steps=100,
                initial_params=initial_params, transforms=transforms)
    mcmc.run(data)
    samples = mcmc.get_samples(num_draws, group_by_chain=group_by_chain)
    # test sample shape
    expected_samples = num_draws if num_draws is not None else num_samples
    if group_by_chain:
        expected_shape = (mcmc.num_chains, expected_samples, 1) if mcmc.num_chains > 1 else (expected_samples, 1)
    else:
        expected_shape = (mcmc.num_chains * expected_samples, 1)
    assert samples['y'].shape == expected_shape

    # test sample stats
    if group_by_chain and mcmc.num_chains > 1:
        samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
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
    mp_context = "spawn" if "CUDA_TEST" in os.environ else None
    with optional(pytest.warns(UserWarning), available_cpu < num_chains):
        mcmc = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=num_chains,
                    initial_params=initial_params, transforms=transforms, mp_context=mp_context)
    mcmc.run(data)
    assert mcmc.num_chains == min(num_chains, available_cpu)
    if mcmc.num_chains == 1:
        assert isinstance(mcmc.sampler, _UnarySampler)
    else:
        assert isinstance(mcmc.sampler, _MultiSampler)


def _empty_model():
    return torch.tensor(1)


def _hook(iters, kernel, samples, stage, i):
    assert samples == {}
    iters.append((stage, i))


@pytest.mark.parametrize("kernel, model", [
    (HMC, _empty_model),
    (NUTS, _empty_model),
])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("num_chains", [
    1,
    skipif_param(2, condition="CI" in os.environ, reason="CI only provides 2-core CPU")
])
def test_null_model_with_hook(kernel, model, jit, num_chains):
    num_warmup, num_samples = 10, 10
    initial_params, potential_fn, transforms, _ = initialize_model(model,
                                                                   num_chains=num_chains)

    iters = []
    hook = partial(_hook, iters)

    mp_context = "spawn" if "CUDA_TEST" in os.environ else None

    kern = kernel(potential_fn=potential_fn, transforms=transforms, jit_compile=jit)
    mcmc = MCMC(kern, num_samples=num_samples, warmup_steps=num_warmup,
                num_chains=num_chains, initial_params=initial_params, hook_fn=hook, mp_context=mp_context)
    mcmc.run()
    samples = mcmc.get_samples()
    assert samples == {}
    if num_chains == 1:
        expected = [("warmup", i) for i in range(num_warmup)] + [("sample", i) for i in range(num_samples)]
        assert iters == expected


@pytest.mark.parametrize("num_chains", [
    1,
    skipif_param(2, condition="CI" in os.environ, reason="CI only provides 2-core CPU")
])
def test_mcmc_diagnostics(num_chains):
    data = torch.tensor([2.0]).repeat(3)
    initial_params, _, transforms, _ = initialize_model(normal_normal_model,
                                                        model_args=(data,),
                                                        num_chains=num_chains)
    kernel = PriorKernel(normal_normal_model)
    mcmc = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=num_chains,
                initial_params=initial_params, transforms=transforms)
    mcmc.run(data)
    diagnostics = mcmc.diagnostics()
    assert diagnostics["y"]["n_eff"].shape == data.shape
    assert diagnostics["y"]["r_hat"].shape == data.shape
    assert diagnostics["dummy_key"] == {'chain {}'.format(i): 'dummy_value'
                                        for i in range(num_chains)}
