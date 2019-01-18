import pytest
import torch

from pyro.distributions import BetaBinomial
from tests.common import assert_equal


@pytest.mark.parametrize("dist", [
    BetaBinomial(2., 5., 10.),
    BetaBinomial(torch.tensor([2., 4.]), torch.tensor([5., 8.]), torch.tensor([10., 12.])),
])
def test_mean(dist):
    analytic_mean = dist.mean
    num_samples = 500000
    sample_mean = dist.sample((num_samples,)).mean(0)
    assert_equal(sample_mean, analytic_mean, prec=0.01)


@pytest.mark.parametrize("dist", [
    BetaBinomial(2., 5., 10.),
    BetaBinomial(torch.tensor([2., 4.]), torch.tensor([5., 8.]), torch.tensor([10., 12.])),
])
def test_variance(dist):
    analytic_var = dist.variance
    num_samples = 500000
    sample_var = dist.sample((num_samples,)).var(0)
    assert_equal(sample_var, analytic_var, prec=0.01)


@pytest.mark.parametrize("dist, values", [
    (BetaBinomial(2., 5., 10.), None),
])
def test_log_prob_support(dist, values):
    if values is None:
        values = dist.enumerate_support()
    log_probs = dist.log_prob(values)
    assert_equal(log_probs.logsumexp(0), torch.tensor(0.), prec=0.01)
