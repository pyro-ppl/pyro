# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

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
from tests.common import assert_close


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
        self.transforms = None

    def setup(self, warmup_steps, data):
        self.data = data
        init_params, potential_fn, transforms, model_trace = initialize_model(self.model,
                                                                              model_args=(data,))
        if self._initial_params is None:
            self._initial_params = init_params
        if self.transforms is None:
            self.transforms = transforms
        self._prototype_trace = model_trace

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
@pytest.mark.filterwarnings("ignore:num_chains")
def test_mcmc_interface(num_draws, group_by_chain, num_chains):
    num_samples = 2000
    data = torch.tensor([1.0])
    initial_params, _, transforms, _ = initialize_model(normal_normal_model, model_args=(data,),
                                                        num_chains=num_chains)
    kernel = PriorKernel(normal_normal_model)
    mcmc = MCMC(kernel=kernel, num_samples=num_samples, warmup_steps=100, num_chains=num_chains,
                mp_context="spawn", initial_params=initial_params, transforms=transforms)
    mcmc.run(data)
    samples = mcmc.get_samples(num_draws, group_by_chain=group_by_chain)
    # test sample shape
    expected_samples = num_draws if num_draws is not None else num_samples
    if group_by_chain:
        expected_shape = (mcmc.num_chains, expected_samples, 1)
    elif num_draws is not None:
        # FIXME: what is the expected behavior of num_draw is not None and group_by_chain=False?
        expected_shape = (expected_samples, 1)
    else:
        expected_shape = (mcmc.num_chains * expected_samples, 1)
    assert samples['y'].shape == expected_shape

    # test sample stats
    if group_by_chain:
        samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
    sample_mean = samples['y'].mean()
    sample_std = samples['y'].std()
    assert_close(sample_mean, torch.tensor(0.0), atol=0.1)
    assert_close(sample_std, torch.tensor(1.0), atol=0.1)


@pytest.mark.parametrize("num_chains, cpu_count", [
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
])
@pytest.mark.parametrize("default_init_params", [True, False])
def test_num_chains(num_chains, cpu_count, default_init_params,
                    monkeypatch):
    monkeypatch.setattr(torch.multiprocessing, 'cpu_count', lambda: cpu_count)
    data = torch.tensor([1.0])
    initial_params, _, transforms, _ = initialize_model(normal_normal_model,
                                                        model_args=(data,),
                                                        num_chains=num_chains)
    if default_init_params:
        initial_params = None
    kernel = PriorKernel(normal_normal_model)
    available_cpu = max(1, cpu_count-1)
    mp_context = "spawn"
    with optional(pytest.warns(UserWarning), available_cpu < num_chains):
        mcmc = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=num_chains,
                    initial_params=initial_params, transforms=transforms, mp_context=mp_context)
    mcmc.run(data)
    assert mcmc.num_chains == num_chains
    if mcmc.num_chains == 1 or available_cpu < num_chains:
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
    2
])
@pytest.mark.filterwarnings("ignore:num_chains")
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
        expected = [("Warmup", i) for i in range(num_warmup)] + [("Sample", i) for i in range(num_samples)]
        assert iters == expected


@pytest.mark.parametrize("num_chains", [
    1,
    2
])
@pytest.mark.filterwarnings("ignore:num_chains")
def test_mcmc_diagnostics(num_chains):
    data = torch.tensor([2.0]).repeat(3)
    initial_params, _, transforms, _ = initialize_model(normal_normal_model,
                                                        model_args=(data,),
                                                        num_chains=num_chains)
    kernel = PriorKernel(normal_normal_model)
    mcmc = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=num_chains, mp_context="spawn",
                initial_params=initial_params, transforms=transforms)
    mcmc.run(data)
    if not torch.backends.mkl.is_available():
        pytest.skip()
    diagnostics = mcmc.diagnostics()
    assert diagnostics["y"]["n_eff"].shape == data.shape
    assert diagnostics["y"]["r_hat"].shape == data.shape
    assert diagnostics["dummy_key"] == {'chain {}'.format(i): 'dummy_value'
                                        for i in range(num_chains)}


@pytest.mark.filterwarnings("ignore:num_chains")
def test_sequential_consistent(monkeypatch):
    # test if there is no stuff left from the previous chain
    monkeypatch.setattr(torch.multiprocessing, 'cpu_count', lambda: 1)

    class FirstKernel(NUTS):
        def setup(self, warmup_steps, *args, **kwargs):
            self._chain_id = 0 if '_chain_id' not in self.__dict__ else 1
            pyro.set_rng_seed(self._chain_id)
            super().setup(warmup_steps, *args, **kwargs)

    class SecondKernel(NUTS):
        def setup(self, warmup_steps, *args, **kwargs):
            self._chain_id = 1 if '_chain_id' not in self.__dict__ else 0
            pyro.set_rng_seed(self._chain_id)
            super().setup(warmup_steps, *args, **kwargs)

    data = torch.tensor([1.0])
    kernel = FirstKernel(normal_normal_model)
    mcmc = MCMC(kernel, num_samples=100, warmup_steps=100, num_chains=2)
    mcmc.run(data)
    samples1 = mcmc.get_samples(group_by_chain=True)

    kernel = SecondKernel(normal_normal_model)
    mcmc = MCMC(kernel, num_samples=100, warmup_steps=100, num_chains=2)
    mcmc.run(data)
    samples2 = mcmc.get_samples(group_by_chain=True)

    assert_close(samples1["y"][0], samples2["y"][1])
    assert_close(samples1["y"][1], samples2["y"][0])
