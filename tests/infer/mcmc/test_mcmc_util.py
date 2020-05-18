# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import (init_to_feasible, init_to_generated, init_to_mean, init_to_median, init_to_sample,
                                  init_to_uniform, init_to_value)
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.util import initialize_model
from pyro.util import optional
from tests.common import assert_close


def beta_bernoulli():
    N = 1000
    true_probs = torch.tensor([0.2, 0.3, 0.4, 0.8, 0.5])
    data = dist.Bernoulli(true_probs).sample([N])

    def model(data=None):
        with pyro.plate("num_components", 5):
            beta = pyro.sample("beta", dist.Beta(1., 1.))
            with pyro.plate("data", N):
                pyro.sample("obs", dist.Bernoulli(beta), obs=data)

    return model, data, true_probs


@pytest.mark.parametrize("num_samples", [100, 200, None])
@pytest.mark.parametrize("parallel", [False, True])
def test_predictive(num_samples, parallel):
    model, data, true_probs = beta_bernoulli()
    init_params, potential_fn, transforms, _ = initialize_model(model,
                                                                model_args=(data,))
    nuts_kernel = NUTS(potential_fn=potential_fn, transforms=transforms)
    mcmc = MCMC(nuts_kernel,
                100,
                initial_params=init_params,
                warmup_steps=100)
    mcmc.run(data)
    samples = mcmc.get_samples()
    with optional(pytest.warns(UserWarning), num_samples not in (None, 100)):
        predictive = Predictive(model, samples,
                                num_samples=num_samples,
                                return_sites=["beta", "obs"],
                                parallel=parallel)
        predictive_samples = predictive()

    # check shapes
    assert predictive_samples["beta"].shape == (100, 1, 5)
    assert predictive_samples["obs"].shape == (100, 1000, 5)

    # check sample mean
    assert_close(predictive_samples["obs"].reshape([-1, 5]).mean(0), true_probs, rtol=0.1)


def model_with_param():
    x = pyro.param("x", torch.tensor(1.))
    pyro.sample("y", dist.Normal(x, 1))


@pytest.mark.parametrize("jit_compile", [False, True])
@pytest.mark.parametrize("num_chains", [1, 2])
@pytest.mark.filterwarnings("ignore:num_chains")
def test_model_with_param(jit_compile, num_chains):
    kernel = NUTS(model_with_param, jit_compile=jit_compile, ignore_jit_warnings=True)
    mcmc = MCMC(kernel, 10, num_chains=num_chains, mp_context="spawn")
    mcmc.run()


@pytest.mark.parametrize("subsample_size", [10, 5])
def test_model_with_subsample(subsample_size):
    size = 10

    def model():
        with pyro.plate("J", size, subsample_size=subsample_size):
            pyro.sample("x", dist.Normal(0, 1))

    kernel = NUTS(model)
    mcmc = MCMC(kernel, 10)
    if subsample_size < size:
        with pytest.raises(RuntimeError, match="subsample"):
            mcmc.run()
    else:
        mcmc.run()


def test_init_to_value():
    def model():
        pyro.sample("x", dist.LogNormal(0, 1))

    value = torch.randn(()).exp() * 10
    kernel = NUTS(model, init_strategy=partial(init_to_value, values={"x": value}))
    kernel.setup(warmup_steps=10)
    assert_close(value, kernel.initial_params['x'].exp())


@pytest.mark.parametrize("init_strategy", [
    init_to_feasible,
    init_to_mean,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
    init_to_feasible(),
    init_to_mean(),
    init_to_median(num_samples=4),
    init_to_sample(),
    init_to_uniform(radius=0.1),
    init_to_value(values={"x": torch.tensor(3.)}),
    init_to_generated(
        generate=lambda: init_to_value(values={"x": torch.rand(())})),
], ids=str)
def test_init_strategy_smoke(init_strategy):
    def model():
        pyro.sample("x", dist.LogNormal(0, 1))

    kernel = NUTS(model, init_strategy=init_strategy)
    kernel.setup(warmup_steps=10)
