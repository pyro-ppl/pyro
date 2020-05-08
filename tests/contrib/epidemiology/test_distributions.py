# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import io

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.epidemiology.distributions import infection_dist, tree_likelihood
from tests.common import assert_close


def assert_dist_close(d1, d2):
    x = torch.arange(float(200))
    p1 = d1.log_prob(x).exp()
    p2 = d2.log_prob(x).exp()

    assert (p1.sum() - 1).abs() < 1e-3, "incomplete mass"
    assert (p2.sum() - 1).abs() < 1e-3, "incomplete mass"

    mean1 = (p1 * x).sum()
    mean2 = (p2 * x).sum()
    assert_close(mean1, mean2, rtol=0.05)

    max_prob = torch.max(p1.max(), p2.max())
    assert (p1 - p2).abs().max() / max_prob < 0.05


@pytest.mark.parametrize("R0,I", [
    (1., 1),
    (1., 10),
    (10., 1),
    (5., 5),
])
def test_binomial_vs_poisson(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=1000., population=1000.)

    assert isinstance(d1, dist.Poisson)
    assert isinstance(d2, dist.Binomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I,k", [
    (1., 1., 0.5),
    (1., 1., 1.),
    (1., 1., 2.),
    (1., 10., 0.5),
    (1., 10., 1.),
    (1., 10., 2.),
    (10., 1., 0.5),
    (10., 1., 1.),
    (10., 1., 2.),
    (5., 5, 0.5),
    (5., 5, 1.),
    (5., 5, 2.),
])
def test_beta_binomial_vs_negative_binomial(R0, I, k):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I, concentration=k)
    d2 = infection_dist(individual_rate=R0, num_infectious=I, concentration=k,
                        num_susceptible=1000., population=1000.)

    assert isinstance(d1, dist.NegativeBinomial)
    assert isinstance(d2, dist.BetaBinomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I", [
    (1., 1.),
    (1., 10.),
    (10., 1.),
    (5., 5.),
])
def test_beta_binomial_vs_binomial(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=20., population=30.)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        num_susceptible=20., population=30.,
                        concentration=200.)

    assert isinstance(d1, dist.Binomial)
    assert isinstance(d2, dist.BetaBinomial)
    assert_dist_close(d1, d2)


@pytest.mark.parametrize("R0,I", [
    (1., 1.),
    (1., 10.),
    (10., 1.),
    (5., 5.),
])
def test_negative_binomial_vs_poisson(R0, I):
    R0 = torch.tensor(R0)
    I = torch.tensor(I)

    d1 = infection_dist(individual_rate=R0, num_infectious=I)
    d2 = infection_dist(individual_rate=R0, num_infectious=I,
                        concentration=200.)

    assert isinstance(d1, dist.Poisson)
    assert isinstance(d2, dist.NegativeBinomial)
    assert_dist_close(d1, d2)


# Loaded from california_tree.nwk
TREE_NWK = (
    "(("
    "EPI_ISL_414648_2020-03-11:0.13583,((("
    "EPI_ISL_417937_2020-03-18:0.05785,"
    "EPI_ISL_417331_2020-03-13:0.04419)"
    "Node_0000001:0.03705,("
    "EPI_ISL_417938_2020-03-18:0.0686,("
    "EPI_ISL_417939_2020-03-18:0.04394,("
    "EPI_ISL_417330_2020-03-13:0.00314,("
    "EPI_ISL_416457_2020-03-18:1e-08,"
    "EPI_ISL_417935_2020-03-18:1e-08)"
    "Node_0000018:0.0168)"
    "Node_0000017:0.02714)"
    "Node_0000016:0.02466)"
    "Node_0000015:0.0263):1e-08,"
    "EPI_ISL_417932_2020-03-18:0.0949)"
    "Node_0000000:0.06006):1e-08,"
    "EPI_ISL_417933_2020-03-18:0.15496)"
    "Node_0000002;"
)


@pytest.fixture
def tree():
    Phylo = pytest.importorskip("Bio.Phylo")
    tree_file = io.StringIO(TREE_NWK)
    trees = list(Phylo.parse(tree_file, "newick"))
    assert len(trees) == 1
    return trees[0]


def test_tree_likelihood(tree):
    R = 1.5
    k = 1.0
    tau = 10.
    I = 1 + torch.arange(20)
    actual = tree_likelihood(R, k, tau, I, tree, time_step=1/365.25)
    assert actual.shape == ()
