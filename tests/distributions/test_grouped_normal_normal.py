# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from pyro.distributions import GroupedNormalNormal, Normal
from tests.common import assert_close


def test_grouped_normal_normal(num_groups=3, num_samples=10**5):
    prior_scale = torch.rand(num_groups)
    prior_loc = torch.randn(num_groups)
    group_idx = torch.cat(
        [torch.arange(num_groups), torch.arange(num_groups), torch.zeros(2).long()]
    )
    values = torch.randn(group_idx.shape)
    obs_scale = torch.rand(group_idx.shape)

    # shape checks
    gnn = GroupedNormalNormal(prior_loc, prior_scale, obs_scale, group_idx)
    assert gnn.log_prob(values).shape == ()
    posterior = gnn.get_posterior(values)
    loc, scale = posterior.loc, posterior.scale
    assert loc.shape == scale.shape == (num_groups,)

    # test correctness of log_prob
    prior_scale = 1 + torch.rand(1).double()
    prior_loc = torch.randn(1).double()
    group_idx = torch.zeros(2).long()
    values = torch.randn(group_idx.shape)
    obs_scale = 0.5 + torch.rand(group_idx.shape).double()

    gnn = GroupedNormalNormal(prior_loc, prior_scale, obs_scale, group_idx)
    actual = gnn.log_prob(values).item()

    prior = Normal(0.0, prior_scale)
    z = prior.sample(sample_shape=(num_samples // 2,))
    z = torch.cat([prior_loc + z, prior_loc - z])
    log_likelihood = Normal(z, obs_scale).log_prob(values).sum(-1)
    expected = torch.logsumexp(log_likelihood, dim=-1).item() - math.log(num_samples)

    assert_close(actual, expected, atol=0.001)
