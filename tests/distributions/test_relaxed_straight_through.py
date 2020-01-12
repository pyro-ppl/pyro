# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints

import pyro
import pyro.optim as optim
from pyro.distributions import (OneHotCategorical, RelaxedBernoulli, RelaxedBernoulliStraightThrough,
                                RelaxedOneHotCategorical, RelaxedOneHotCategoricalStraightThrough)
from pyro.infer import SVI, Trace_ELBO
from tests.common import assert_equal

ONEHOT_PROBS = [
                [0.25, 0.75],
                [0.25, 0.5, 0.25],
                [[0.25, 0.75], [0.75, 0.25]],
                [[[0.25, 0.75]], [[0.75, 0.25]]],
                [0.1] * 10,
]

BERN_PROBS = [
               [0.25, 0.75],
               [[0.25, 0.75], [0.75, 0.25]]
]


@pytest.mark.parametrize('probs', ONEHOT_PROBS)
def test_onehot_shapes(probs):
    temperature = torch.tensor(0.5)
    probs = torch.tensor(probs, requires_grad=True)
    d = RelaxedOneHotCategoricalStraightThrough(temperature, probs=probs)
    sample = d.rsample()
    log_prob = d.log_prob(sample)
    grad_probs = grad(log_prob.sum(), [probs])[0]
    assert grad_probs.shape == probs.shape


@pytest.mark.parametrize('temp', [0.3, 0.5, 1.0])
def test_onehot_entropy_grad(temp):
    num_samples = 2000000
    q = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    temp = torch.tensor(temp)

    dist_q = RelaxedOneHotCategorical(temperature=temp, probs=q)
    z = dist_q.rsample(sample_shape=(num_samples,))
    expected = grad(dist_q.log_prob(z).sum(), [q])[0] / num_samples

    dist_q = RelaxedOneHotCategoricalStraightThrough(temperature=temp, probs=q)
    z = dist_q.rsample(sample_shape=(num_samples,))
    actual = grad(dist_q.log_prob(z).sum(), [q])[0] / num_samples

    assert_equal(expected, actual, prec=0.08,
                 msg='bad grad for RelaxedOneHotCategoricalStraightThrough (expected {}, got {})'.
                 format(expected, actual))


def test_onehot_svi_usage():

    def model():
        p = torch.tensor([0.25] * 4)
        pyro.sample('z', OneHotCategorical(probs=p))

    def guide():
        q = pyro.param('q', torch.tensor([0.1, 0.2, 0.3, 0.4]), constraint=constraints.simplex)
        temp = torch.tensor(0.10)
        pyro.sample('z', RelaxedOneHotCategoricalStraightThrough(temperature=temp, probs=q))

    adam = optim.Adam({"lr": .001, "betas": (0.95, 0.999)})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for k in range(6000):
        svi.step()

    assert_equal(pyro.param('q'), torch.tensor([0.25] * 4), prec=0.01,
                 msg='test svi usage of RelaxedOneHotCategoricalStraightThrough failed')


@pytest.mark.parametrize('probs', BERN_PROBS)
def test_bernoulli_shapes(probs):
    temperature = torch.tensor(0.5)
    probs = torch.tensor(probs, requires_grad=True)
    d = RelaxedBernoulliStraightThrough(temperature, probs=probs)
    sample = d.rsample()
    log_prob = d.log_prob(sample)
    grad_probs = grad(log_prob.sum(), [probs])[0]
    assert grad_probs.shape == probs.shape


@pytest.mark.parametrize('temp', [0.5, 1.0])
def test_bernoulli_entropy_grad(temp):
    num_samples = 1500000
    q = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    temp = torch.tensor(temp)

    dist_q = RelaxedBernoulli(temperature=temp, probs=q)
    z = dist_q.rsample(sample_shape=(num_samples,))
    expected = grad(dist_q.log_prob(z).sum(), [q])[0] / num_samples

    dist_q = RelaxedBernoulliStraightThrough(temperature=temp, probs=q)
    z = dist_q.rsample(sample_shape=(num_samples,))
    actual = grad(dist_q.log_prob(z).sum(), [q])[0] / num_samples

    assert_equal(expected, actual, prec=0.04,
                 msg='bad grad for RelaxedBernoulliStraightThrough (expected {}, got {})'.
                 format(expected, actual))
