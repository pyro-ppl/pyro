# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import io
import re

import pytest
import torch

import pyro
from pyro.distributions import CoalescentTimes, CoalescentTimesWithRate
from pyro.distributions.coalescent import (CoalescentRateLikelihood, CoalescentTimesConstraint,
                                           _sample_coalescent_times, bio_phylo_to_times)
from pyro.distributions.util import broadcast_shape
from tests.common import assert_close


@pytest.mark.parametrize("num_leaves", range(2, 30))
def test_sample_is_valid(num_leaves):
    pyro.set_rng_seed(num_leaves)

    # Check with disperse leaves.
    leaf_times = torch.randn(num_leaves)
    coal_times = _sample_coalescent_times(leaf_times)
    assert CoalescentTimesConstraint(leaf_times).check(coal_times)
    assert len(set(coal_times.tolist())) == len(coal_times)

    # Check with simultaneous leaves.
    leaf_times = torch.zeros(num_leaves)
    coal_times = _sample_coalescent_times(leaf_times)
    assert CoalescentTimesConstraint(leaf_times).check(coal_times)
    assert len(set(coal_times.tolist())) == len(coal_times)


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (7,), (4, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (6,), (2, 3)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 3, 5, 11])
def test_simple_smoke(num_leaves, num_steps, batch_shape, sample_shape):
    leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
    d = CoalescentTimes(leaf_times)
    coal_times = d.sample(sample_shape)
    assert coal_times.shape == sample_shape + batch_shape + (num_leaves-1,)

    actual = d.log_prob(coal_times)
    assert actual.shape == sample_shape + batch_shape


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (6,), (4, 5)], ids=str)
@pytest.mark.parametrize("rate_grid_shape", [(), (2,), (3, 1), (3, 2)], ids=str)
@pytest.mark.parametrize("leaf_times_shape", [(), (2,), (3, 1), (3, 2)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 7, 11])
def test_with_rate_smoke(num_leaves, num_steps, leaf_times_shape, rate_grid_shape, sample_shape):
    batch_shape = broadcast_shape(leaf_times_shape, rate_grid_shape)
    leaf_times = torch.rand(leaf_times_shape + (num_leaves,)).pow(0.5) * num_steps
    rate_grid = torch.rand(rate_grid_shape + (num_steps,))
    d = CoalescentTimesWithRate(leaf_times, rate_grid)
    coal_times = _sample_coalescent_times(
        leaf_times.expand(sample_shape + batch_shape + (-1,)))
    assert coal_times.shape == sample_shape + batch_shape + (num_leaves-1,)

    actual = d.log_prob(coal_times)
    assert actual.shape == sample_shape + batch_shape


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (5,)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 7, 11])
def test_log_prob_unit_rate(num_leaves, num_steps, batch_shape, sample_shape):
    leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
    d1 = CoalescentTimes(leaf_times)

    rate_grid = torch.ones(batch_shape + (num_steps,))
    d2 = CoalescentTimesWithRate(leaf_times, rate_grid)

    coal_times = d1.sample(sample_shape)
    assert_close(d1.log_prob(coal_times), d2.log_prob(coal_times))


@pytest.mark.parametrize("num_steps", [9])
@pytest.mark.parametrize("sample_shape", [(), (5,)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("num_leaves", [2, 7, 11])
def test_log_prob_constant_rate(num_leaves, num_steps, batch_shape, sample_shape):
    rate = torch.randn(batch_shape).exp()
    rate_grid = rate.unsqueeze(-1).expand(batch_shape + (num_steps,))
    leaf_times_2 = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
    leaf_times_1 = leaf_times_2 * rate.unsqueeze(-1)

    d1 = CoalescentTimes(leaf_times_1)
    coal_times_1 = d1.sample(sample_shape)
    log_prob_1 = d1.log_prob(coal_times_1)

    d2 = CoalescentTimesWithRate(leaf_times_2, rate_grid)
    coal_times_2 = coal_times_1 / rate.unsqueeze(-1)
    log_prob_2 = d2.log_prob(coal_times_2)

    log_abs_det_jacobian = -coal_times_2.size(-1) * rate.log()
    assert_close(log_prob_1 - log_abs_det_jacobian, log_prob_2)


@pytest.mark.parametrize("clamped", [True, False], ids=["clamped", "unclamped"])
@pytest.mark.parametrize("num_steps", [2, 5, 10, 20])
@pytest.mark.parametrize("num_leaves", [2, 5, 10, 20])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_likelihood_vectorized(num_leaves, num_steps, batch_shape, clamped):
    if clamped:
        leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
        coal_times = CoalescentTimes(leaf_times).sample().clamp(min=0)
    else:
        leaf_times = torch.randn(batch_shape + (num_leaves,))
        leaf_times.mul_(0.25).add_(0.75).mul_(num_steps)
        coal_times = CoalescentTimes(leaf_times).sample()

    rate_grid = torch.rand(batch_shape + (num_steps,)) + 0.5

    d = CoalescentTimesWithRate(leaf_times, rate_grid)
    expected = d.log_prob(coal_times)

    likelihood = CoalescentRateLikelihood(leaf_times, coal_times, num_steps)
    actual = likelihood(rate_grid).sum(-1)

    assert_close(actual, expected)


@pytest.mark.parametrize("clamped", [True, False], ids=["clamped", "unclamped"])
@pytest.mark.parametrize("num_steps", [2, 5, 10, 20])
@pytest.mark.parametrize("num_leaves", [2, 5, 10, 20])
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
def test_likelihood_sequential(num_leaves, num_steps, batch_shape, clamped):
    if clamped:
        leaf_times = torch.rand(batch_shape + (num_leaves,)).pow(0.5) * num_steps
        coal_times = CoalescentTimes(leaf_times).sample().clamp(min=0)
    else:
        leaf_times = torch.randn(batch_shape + (num_leaves,))
        leaf_times.mul_(0.25).add_(0.75).mul_(num_steps)
        coal_times = CoalescentTimes(leaf_times).sample()

    rate_grid = torch.rand(batch_shape + (num_steps,)) + 0.5

    d = CoalescentTimesWithRate(leaf_times, rate_grid)
    expected = d.log_prob(coal_times)

    likelihood = CoalescentRateLikelihood(leaf_times, coal_times, num_steps)
    actual = sum(likelihood(rate_grid[..., t], t)
                 for t in range(num_steps))

    assert_close(actual, expected)


TREE_NEXUS = """
#NEXUS
Begin Trees;
 Tree tree1=((EPI_ISL_408009:0.00000[&date=2020.08],
 EPI_ISL_408008:0.00000[&date=2020.08]) NODE_0000004:0.17430[&date=2020.08],
 (EPI_ISL_417931:0.28554[&date=2020.21],
 (EPI_ISL_417332:0.11102[&date=2020.20], EPI_ISL_413931:0.08643[&date=2020.18])
 NODE_0000005:0.16360[&date=2020.09], ((EPI_ISL_413558:0.11909[&date=2020.16],
     (EPI_ISL_413559:0.07179[&date=2020.16],
         (EPI_ISL_412862:0.00000[&date=2020.15],
             EPI_ISL_413561:0.01093[&date=2020.16])
         NODE_0000011:0.06086[&date=2020.15])
     NODE_0000012:0.04730[&date=2020.09]) NODE_0000007:0.06603[&date=2020.04],
     (EPI_ISL_411955:0.09393[&date=2020.11],
         (EPI_ISL_417325:0.08372[&date=2020.17],
             (EPI_ISL_417318:0.02411[&date=2020.16],
                 EPI_ISL_417320:0.03504[&date=2020.17])
             NODE_0000009:0.05141[&date=2020.14])
         NODE_0000006:0.07032[&date=2020.09])
     NODE_0000014:0.04474[&date=2020.02]) NODE_0000010:0.04578[&date=2019.97],
 (EPI_ISL_417933:0.15496[&date=2020.21], EPI_ISL_414648:0.13583[&date=2020.19],
         (EPI_ISL_417932:0.09490[&date=2020.21],
             (EPI_ISL_417937:0.05785[&date=2020.21],
                 EPI_ISL_417331:0.04419[&date=2020.20])
             NODE_0000001:0.03705[&date=2020.15],
             (EPI_ISL_417938:0.06860[&date=2020.21],
                 (EPI_ISL_417939:0.04394[&date=2020.21],
                     (EPI_ISL_417330:0.00314[&date=2020.20],
                         (EPI_ISL_416457:0.00000[&date=2020.21],
                             EPI_ISL_417935:0.00000[&date=2020.21])
                         NODE_0000018:0.01680[&date=2020.21])
                     NODE_0000017:0.02714[&date=2020.19])
                 NODE_0000016:0.02466[&date=2020.17])
                 NODE_0000015:0.02630[&date=2020.14])
                 NODE_0000000:0.06006[&date=2020.12])
         NODE_0000002:0.13059[&date=2020.06])
 NODE_0000003:0.02264[&date=2019.93]) NODE_0000008:0.10000[&date=2019.90];
End;
"""


@pytest.fixture
def tree():
    Phylo = pytest.importorskip("Bio.Phylo")
    tree_file = io.StringIO(TREE_NEXUS)
    trees = list(Phylo.parse(tree_file, "nexus"))
    assert len(trees) == 1
    return trees[0]


def test_bio_phylo_to_times(tree):
    leaf_times, coal_times = bio_phylo_to_times(tree)
    assert len(coal_times) + 1 == len(leaf_times)

    # Check positivity.
    times = torch.cat([coal_times, leaf_times])
    signs = torch.cat([-torch.ones_like(coal_times), torch.ones_like(leaf_times)])
    times, index = times.sort(0)
    signs = signs[index]
    lineages = signs.flip([0]).cumsum(0).flip([0])
    assert (lineages >= 0).all()


def test_bio_phylo_to_times_custom(tree):
    # Test a custom time parser.
    def get_time(clade):
        date_string = re.search(r"date=(\d\d\d\d\.\d\d)", clade.comment).group(1)
        return (float(date_string) - 2020) * 365.25

    leaf_times, coal_times = bio_phylo_to_times(tree, get_time=get_time)
    assert len(coal_times) + 1 == len(leaf_times)

    # Check positivity.
    times = torch.cat([coal_times, leaf_times])
    signs = torch.cat([-torch.ones_like(coal_times), torch.ones_like(leaf_times)])
    times, index = times.sort(0)
    signs = signs[index]
    lineages = signs.flip([0]).cumsum(0).flip([0])
    assert (lineages >= 0).all()
