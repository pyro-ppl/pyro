import torch

import pytest

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, HMC, NUTS
import pyro.poutine as poutine


def assert_ok(mcmc_kernel):
    """
    Assert that inference works without warnings or errors.
    """
    MCMC(mcmc_kernel, num_samples=10, warmup_steps=10).run()


def assert_error(mcmc_kernel):
    """
    Assert that inference fails with an error.
    """
    with pytest.raises(ValueError):
        MCMC(mcmc_kernel, num_samples=10, warmup_steps=10).run()


@pytest.mark.parametrize("kernel, kwargs", [
    (HMC, {"adapt_step_size": True, "num_steps": 3}),
    (NUTS, {"adapt_step_size": True}),
])
def test_model_error_stray_batch_dims(kernel, kwargs):
    @poutine.broadcast
    def gmm():
        data = torch.tensor([0., 0., 3., 3., 3., 5., 5.])
        mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.ones(3)))
        cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(3), 1.))
        with pyro.iarange("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    mcmc_kernel = kernel(gmm, **kwargs)
    # Error due to non finite value for `max_iarange_nesting`.
    assert_error(mcmc_kernel)
    # Error due to batch dims not inside iarange.
    mcmc_kernel = kernel(gmm, max_iarange_nesting=1, **kwargs)
    assert_error(mcmc_kernel)
    # No error with validation disabled.
    with pyro.validation_enabled(False):
        assert_ok(mcmc_kernel)


@pytest.mark.parametrize("kernel, kwargs", [
    (HMC, {"adapt_step_size": True, "num_steps": 3}),
    (NUTS, {"adapt_step_size": True}),
])
def test_model_error_enum_dim_clash(kernel, kwargs):
    @poutine.broadcast
    def gmm():
        data = torch.tensor([0., 0., 3., 3., 3., 5., 5.])
        with pyro.iarange("num_clusters", 3):
            mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.tensor(1.)))
            cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(3), 1.))
        with pyro.iarange("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    mcmc_kernel = kernel(gmm, max_iarange_nesting=0, **kwargs)
    assert_error(mcmc_kernel)
