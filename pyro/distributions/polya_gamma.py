# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints

from pyro.distributions.torch import Exponential
from pyro.distributions.torch_distribution import TorchDistribution


class TruncatedPolyaGamma(TorchDistribution):
    """
    This is a PolyaGamma(1, 0) distribution truncated to have finite support in
    the interval (0, 2.5). See [1] for details. As a consequence of the truncation
    the `log_prob` method is only accurate to about six decimal places. In
    addition the provided sampler is a rough approximation that is only meant to
    be used in contexts where sample accuracy is not important (e.g. in initialization).
    Broadly, this implementation is only intended for usage in cases where good
    approximations of the `log_prob` are sufficient, as is the case e.g. in HMC.

    :param tensor prototype: A prototype tensor of arbitrary shape used to determine
        the `dtype` and `device` returned by `sample` and `log_prob`.

    References

    [1] 'Bayesian inference for logistic models using Polya-Gamma latent variables'
        Nicholas G. Polson, James G. Scott, Jesse Windle.
    """
    truncation_point = 2.5
    num_log_prob_terms = 7
    num_gamma_variates = 8
    assert num_log_prob_terms % 2 == 1

    arg_constraints = {}
    support = constraints.interval(0.0, truncation_point)
    has_rsample = False

    def __init__(self, prototype, validate_args=None):
        self.prototype = prototype
        super(TruncatedPolyaGamma, self).__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TruncatedPolyaGamma, _instance)
        super(TruncatedPolyaGamma, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get("_validate_args")
        new.prototype = self.prototype
        return new

    def sample(self, sample_shape=()):
        denom = torch.arange(0.5, self.num_gamma_variates, device=self.prototype.device).pow(2.0)
        ones = self.prototype.new_ones((self.num_gamma_variates))
        x = Exponential(ones).sample(self.batch_shape + sample_shape)
        x = (x / denom).sum(-1)
        return torch.clamp(x * (0.5 / math.pi ** 2), max=self.truncation_point)

    def log_prob(self, value):
        value = value.unsqueeze(-1)
        two_n_plus_one = 2.0 * torch.arange(0, self.num_log_prob_terms, device=self.prototype.device) + 1.0
        log_terms = two_n_plus_one.log() - 1.5 * value.log() - 0.125 * two_n_plus_one.pow(2.0) / value
        even_terms = log_terms[..., ::2]
        odd_terms = log_terms[..., 1::2]
        sum_even = torch.logsumexp(even_terms, dim=-1).exp()
        sum_odd = torch.logsumexp(odd_terms, dim=-1).exp()
        return (sum_even - sum_odd).log() - 0.5 * math.log(2.0 * math.pi)
