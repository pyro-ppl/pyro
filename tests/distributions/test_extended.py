# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from torch.autograd import grad

import pyro.distributions as dist
from pyro.contrib.epidemiology.distributions import set_approx_log_prob_tol
from tests.common import assert_equal


def check_grad(value, *params):
    grads = grad(value.sum(), params, create_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("tol", [0., 0.02, 0.05, 0.1])
def test_extended_binomial(tol):
    with set_approx_log_prob_tol(tol):
        total_count = torch.tensor([0., 1., 2., 10.])
        probs = torch.tensor([0.5, 0.5, 0.4, 0.2]).requires_grad_()

        d1 = dist.Binomial(total_count, probs)
        d2 = dist.ExtendedBinomial(total_count, probs)
        # Check on good data.
        data = d1.sample((100,))
        assert_equal(d1.log_prob(data), d2.log_prob(data))

        # Check on extended data.
        data = torch.arange(-10., 20.).unsqueeze(-1)
        with pytest.raises(ValueError):
            d1.log_prob(data)
        log_prob = d2.log_prob(data)
        valid = d1.support.check(data)
        assert ((log_prob > -math.inf) == valid).all()
        check_grad(log_prob, probs)

        # Check on shape error.
        with pytest.raises(ValueError):
            d2.log_prob(torch.tensor([0., 0.]))

        # Check on value error.
        with pytest.raises(ValueError):
            d2.log_prob(torch.tensor(0.5))

        # Check on negative total_count.
        total_count = torch.arange(-10, 0.)
        probs = torch.tensor(0.5).requires_grad_()
        d = dist.ExtendedBinomial(total_count, probs)
        log_prob = d.log_prob(data)
        assert (log_prob == -math.inf).all()
        check_grad(log_prob, probs)


@pytest.mark.parametrize("tol", [0., 0.02, 0.05, 0.1])
def test_extended_beta_binomial(tol):
    with set_approx_log_prob_tol(tol):
        concentration1 = torch.tensor([0.2, 1.0, 2.0, 1.0]).requires_grad_()
        concentration0 = torch.tensor([0.2, 0.5, 1.0, 2.0]).requires_grad_()
        total_count = torch.tensor([0., 1., 2., 10.])

        d1 = dist.BetaBinomial(concentration1, concentration0, total_count)
        d2 = dist.ExtendedBetaBinomial(concentration1, concentration0, total_count)

        # Check on good data.
        data = d1.sample((100,))
        assert_equal(d1.log_prob(data), d2.log_prob(data))

        # Check on extended data.
        data = torch.arange(-10., 20.).unsqueeze(-1)
        with pytest.raises(ValueError):
            d1.log_prob(data)
        log_prob = d2.log_prob(data)
        valid = d1.support.check(data)
        assert ((log_prob > -math.inf) == valid).all()
        check_grad(log_prob, concentration1, concentration0)

        # Check on shape error.
        with pytest.raises(ValueError):
            d2.log_prob(torch.tensor([0., 0.]))

        # Check on value error.
        with pytest.raises(ValueError):
            d2.log_prob(torch.tensor(0.5))

        # Check on negative total_count.
        concentration1 = torch.tensor(1.5).requires_grad_()
        concentration0 = torch.tensor(1.5).requires_grad_()
        total_count = torch.arange(-10, 0.)
        d = dist.ExtendedBetaBinomial(concentration1, concentration0, total_count)
        log_prob = d.log_prob(data)
        assert (log_prob == -math.inf).all()
        check_grad(log_prob, concentration1, concentration0)
