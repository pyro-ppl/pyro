# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions.utils import broadcast_all

from pyro.distributions import Normal, constraints
from pyro.distributions.torch_distribution import TorchDistribution

LOG_ROOT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


class GroupedNormalNormal(TorchDistribution):
    r"""
    This likelihood, which operates on groups of real-valued scalar observations, is obtained by
    integrating out a latent mean for each group. Both the prior on each latent mean as well as the
    observation likelihood for each data point are univariate Normal distributions.
    The prior means are controlled by `prior_loc` and `prior_scale`. The observation noise of the
    Normal likelihood is controlled by `obs_scale`, which is allowed to vary from observation to
    observation. The tensor of indices `group_idx` connects each observation to one of the groups
    specified by `prior_loc` and `prior_scale`.

    See e.g. Eqn. (55) in ref. [1] for relevant expressions in a simpler case with scalar `obs_scale`.

    References:
    [1] "Conjugate Bayesian analysis of the Gaussian distribution," Kevin P. Murphy.

    Args:
        prior_loc: (num_groups,)
        prior_scale: (num_groups,)
        obs_scale: (num_data,)
        group_idx: (num_data,)
        value: (num_data,)
    """
    arg_constraints = {
        "prior_loc": constraints.real,
        "prior_scale": constraints.positive,
        "obs_scale": constraints.positive,
    }
    support = constraints.real

    def __init__(
        self, prior_loc, prior_scale, obs_scale, group_idx, validate_args=None
    ):
        prior_loc, prior_scale = broadcast_all(prior_loc, prior_scale)
        obs_scale, group_idx = broadcast_all(obs_scale, group_idx)

        self.prior_loc = prior_loc
        self.prior_scale = prior_scale
        self.obs_scale = obs_scale
        self.group_idx = group_idx
        batch_shape = prior_loc.shape[:-1]

        if batch_shape != torch.Size([]):
            raise ValueError("GroupedNormalNormal only supports trivial batch_shape's.")

        self.num_groups = prior_loc.size(0)
        if group_idx.min().item() < 0 or group_idx.max().item() >= self.num_groups:
            raise ValueError(
                "Each index in group_idx must be an integer in the inclusive range [0, prior_loc.size(0) - 1]."
            )

        self.num_data_per_batch = prior_loc.new_zeros(self.num_groups).scatter_add(
            0, self.group_idx, prior_loc.new_ones(self.group_idx.shape)
        )
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    def sample(self, sample_shape=()):
        raise NotImplementedError

    def get_posterior(self, value):
        obs_scale_sq_inv = self.obs_scale.pow(-2)
        prior_scale_sq_inv = self.prior_scale.pow(-2)

        obs_scale_sq_inv_sum = torch.zeros_like(self.prior_loc).scatter_add(
            0, self.group_idx, obs_scale_sq_inv
        )
        precision = prior_scale_sq_inv + obs_scale_sq_inv_sum
        scaled_value_sum = torch.zeros_like(self.prior_loc).scatter_add(
            0, self.group_idx, value * obs_scale_sq_inv
        )

        loc = (scaled_value_sum + self.prior_loc * prior_scale_sq_inv) / precision
        scale = precision.pow(-0.5)

        return Normal(loc=loc, scale=scale)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        group_idx = self.group_idx

        if value.shape != group_idx.shape:
            raise ValueError(
                "GroupedNormalNormal.log_prob only supports values that have the same shape as group_idx."
            )

        prior_scale_sq = self.prior_scale.pow(2.0)
        obs_scale_sq_inv = self.obs_scale.pow(-2)
        obs_scale_sq_inv_sum = torch.zeros_like(self.prior_loc).scatter_add(
            0, self.group_idx, obs_scale_sq_inv
        )

        scale_ratio = prior_scale_sq * obs_scale_sq_inv_sum
        delta = value - self.prior_loc[group_idx]
        scaled_delta = delta * obs_scale_sq_inv
        scaled_delta_sum = torch.zeros_like(self.prior_loc).scatter_add(
            0, self.group_idx, scaled_delta
        )

        result1 = -(self.num_data_per_batch * LOG_ROOT_TWO_PI).sum()
        result2 = -0.5 * torch.log1p(scale_ratio).sum() - self.obs_scale.log().sum()
        result3 = -0.5 * torch.dot(delta, scaled_delta)
        numerator = prior_scale_sq * scaled_delta_sum.pow(2)
        result4 = 0.5 * (numerator / (1.0 + scale_ratio)).sum()

        return result1 + result2 + result3 + result4
