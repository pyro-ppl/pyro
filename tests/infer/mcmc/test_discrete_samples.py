# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import HMC, MCMC, NUTS
from tests.common import assert_close, assert_equal


@pytest.mark.parametrize("include_discrete", [True, False])
def test_categorical_only_hmc(include_discrete):
    """Test HMC with only categorical variables (issue #3368)"""

    def model(oil_pr):
        oil = pyro.sample("oil", dist.Categorical(oil_pr))
        seis_dist = torch.tensor(
            [[0.1, 0.3, 0.6], [0.3, 0.4, 0.3], [0.5, 0.4, 0.1]]
        )
        seis = pyro.sample("seis", dist.Categorical(seis_dist[oil, :]))
        return seis

    pyro.clear_param_store()
    conditioned_model = pyro.condition(model, data={"seis": torch.tensor(1)})
    hmc_kernel = HMC(conditioned_model, step_size=0.9, num_steps=4)
    posterior = MCMC(hmc_kernel, num_samples=10, warmup_steps=50)
    posterior.run(torch.tensor([1 / 3, 1 / 3, 1 / 3]))

    samples = posterior.get_samples(include_discrete=include_discrete)

    if include_discrete:
        assert "oil" in samples, "Oil samples should be present when include_discrete=True"
        assert samples["oil"].shape == torch.Size([10])
        assert torch.all((samples["oil"] >= 0) & (samples["oil"] <= 2))
    else:
        assert (
            len(samples) == 0
        ), "No samples expected when include_discrete=False and no continuous vars"


@pytest.mark.parametrize("kernel_cls", [HMC, NUTS])
def test_mixed_continuous_discrete(kernel_cls):
    """Test with both continuous and discrete variables"""

    def model(data):
        p = pyro.sample("p", dist.Beta(2.0, 2.0))
        z = pyro.sample("z", dist.Categorical(torch.tensor([0.3, 0.7])))
        means = torch.tensor([0.0, 1.0])
        pyro.sample("obs", dist.Normal(means[z], 1.0), obs=data)
        return z

    pyro.clear_param_store()
    data = torch.tensor(0.9)
    if kernel_cls == HMC:
        kernel = kernel_cls(model, step_size=0.1, num_steps=10)
    else:
        kernel = kernel_cls(model)
    posterior = MCMC(kernel, num_samples=20, warmup_steps=50)
    posterior.run(data)

    samples = posterior.get_samples()

    assert "p" in samples, "Continuous variable p should be present"
    assert "z" in samples, "Discrete variable z should be present"
    assert samples["p"].shape == torch.Size([20])
    assert samples["z"].shape == torch.Size([20])
    assert torch.all((samples["z"] >= 0) & (samples["z"] <= 1))

    # Test that include_discrete=False excludes discrete samples
    samples_no_discrete = posterior.get_samples(include_discrete=False)
    assert "p" in samples_no_discrete
    assert "z" not in samples_no_discrete


def test_group_by_chain_with_discrete():
    """Test group_by_chain parameter with discrete variables"""

    def model(data):
        z = pyro.sample("z", dist.Categorical(torch.tensor([0.5, 0.5])))
        means = torch.tensor([0.0, 1.0])
        pyro.sample("obs", dist.Normal(means[z], 1.0), obs=data)

    pyro.clear_param_store()
    data = torch.tensor(0.8)
    nuts_kernel = NUTS(model)
    posterior = MCMC(nuts_kernel, num_samples=10, warmup_steps=20, num_chains=1)
    posterior.run(data)

    samples_grouped = posterior.get_samples(group_by_chain=True)
    assert "z" in samples_grouped
    assert samples_grouped["z"].shape == torch.Size([1, 10])

    samples_flat = posterior.get_samples(group_by_chain=False)
    assert samples_flat["z"].shape == torch.Size([10])


def test_discrete_with_plate():
    """Test discrete sampling with plate"""

    def model(data):
        with pyro.plate("data", len(data)):
            z = pyro.sample("z", dist.Categorical(torch.tensor([0.5, 0.5])))
            means = torch.tensor([0.0, 1.0])
            pyro.sample("obs", dist.Normal(means[z], 1.0), obs=data)

    pyro.clear_param_store()
    data = torch.tensor([0.9, 1.1, 0.8])
    nuts_kernel = NUTS(model, max_plate_nesting=1)
    posterior = MCMC(nuts_kernel, num_samples=15, warmup_steps=30)
    posterior.run(data)

    samples = posterior.get_samples()
    assert "z" in samples
    # Each sample should have 3 discrete values (one per data point)
    assert samples["z"].shape == torch.Size([15, 3])
    assert torch.all((samples["z"] >= 0) & (samples["z"] <= 1))
