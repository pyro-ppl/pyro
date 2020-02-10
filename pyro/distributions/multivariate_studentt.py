# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch import Chi2
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class MultivariateStudentT(TorchDistribution):
    """
    Creates a multivariate Student's t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale_tril`.

    :param ~torch.Tensor df: degrees of freedom
    :param ~torch.Tensor loc: mean of the distribution
    :param ~torch.Tensor scale_tril: scale of the distribution, which is
        a lower triangular matrix with positive diagonal entries
    """
    arg_constraints = {'df': constraints.positive,
                       'loc': constraints.real_vector,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, df, loc, scale_tril, validate_args=None):
        dim = loc.size(-1)
        assert scale_tril.shape[-2:] == (dim, dim)
        if not isinstance(df, torch.Tensor):
            df = loc.new_tensor(df)
        batch_shape = broadcast_shape(df.shape, loc.shape[:-1], scale_tril.shape[:-2])
        event_shape = (dim,)
        self.df = df.expand(batch_shape)
        self.loc = loc.expand(batch_shape + event_shape)
        self._unbroadcasted_scale_tril = scale_tril
        self._chi2 = Chi2(self.df)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        # NB: this is not covariance of this distribution;
        # the actual covariance is df / (df - 2) * covariance_matrix
        return (torch.matmul(self._unbroadcasted_scale_tril,
                             self._unbroadcasted_scale_tril.transpose(-1, -2))
                .expand(self._batch_shape + self._event_shape + self._event_shape))

    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(self.loc.size(-1), device=self.loc.device, dtype=self.loc.dtype)
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        scale_shape = loc_shape + self.event_shape
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(scale_shape)
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(scale_shape)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(scale_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(MultivariateStudentT, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        X = torch.empty(shape, dtype=self.df.dtype, device=self.df.device).normal_()
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df).unsqueeze(-1)
        return self.loc + self.scale_tril.matmul(Y.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        n = self.loc.size(-1)
        y = (value - self.loc).unsqueeze(-1).triangular_solve(self.scale_tril, upper=False).solution.squeeze(-1)
        Z = (self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) +
             0.5 * n * self.df.log() +
             0.5 * n * math.log(math.pi) +
             torch.lgamma(0.5 * self.df) -
             torch.lgamma(0.5 * (self.df + n)))
        return -0.5 * (self.df + n) * torch.log1p(y.pow(2).sum(-1) / self.df) - Z

    @property
    def mean(self):
        m = self.loc.clone()
        m[self.df <= 1, :] = float('nan')
        return m

    @property
    def variance(self):
        m = self.scale_tril.pow(2).sum(-1) * (self.df / (self.df - 2)).unsqueeze(-1)
        m[(self.df <= 2) & (self.df > 1), :] = float('inf')
        m[self.df <= 1, :] = float('nan')
        return m
