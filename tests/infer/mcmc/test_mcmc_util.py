# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
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
