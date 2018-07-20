import torch

import pytest

import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro.infer.mcmc import MCMC, HMC, NUTS
import pyro.poutine as poutine
from tests.common import assert_equal


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


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-1.3434)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-4.1813))
])
def test_log_prob_computation(data, expected_log_prob):
    @poutine.broadcast
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        with pyro.iarange("data", len(data)):
            z = pyro.sample("z", dist.Bernoulli(q))
            mean = 2 * z - 1
            pyro.sample("obs", dist.Normal(mean, 1.), obs=data)

    hmc_kernel = HMC(model, adapt_step_size=True, num_steps=3, max_iarange_nesting=1)
    hmc_kernel._has_enumerable_sites = True
    conditioned_model = poutine.enum(
        config_enumerate(poutine.condition(model, {"p": torch.tensor(0.4)}),
                         default="parallel"),
        first_available_dim=1)
    model_trace = poutine.trace(conditioned_model).get_trace(data)
    assert_equal(hmc_kernel._compute_trace_log_prob(model_trace),
                 expected_log_prob,
                 prec=1e-3)
