from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints, kl_divergence, register_kl
from torch.distributions.multivariate_normal import _batch_mv
from torch.distributions.lowrank_multivariate_normal import (_batch_lowrank_mahalanobis,
                                                             _batch_lowrank_logdet)
from torch.distributions.utils import _standard_normal

from pyro.distributions.torch_distribution import IndependentConstraint, TorchDistributionMixin
from pyro.distributions.util import sum_rightmost


class LowRankMultivariateNormal(torch.distributions.LowRankMultivariateNormal,
                                TorchDistributionMixin):
    # Resolve the issue https://github.com/uber/pyro/issues/1586
    def __init__(self, loc, cov_factor, cov_diag, validate_args=None):
        super(LowRankMultivariateNormal, self).__init__(loc, cov_factor, cov_diag, validate_args)
        self._unbroadcasted_cov_factor = cov_factor
        self._unbroadcasted_cov_diag = cov_diag

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LowRankMultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new.cov_diag = self.cov_diag.expand(loc_shape)
        new.cov_factor = self.cov_factor.expand(loc_shape + self.cov_factor.shape[-1:])
        new._unbroadcasted_cov_factor = self._unbroadcasted_cov_factor
        new._unbroadcasted_cov_diag = self._unbroadcasted_cov_diag
        new._capacitance_tril = self._capacitance_tril
        super(torch.distributions.LowRankMultivariateNormal, new).__init__(batch_shape,
                                                                           self.event_shape,
                                                                           validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        W_shape = shape[:-1] + self.cov_factor.shape[-1:]
        eps_W = _standard_normal(W_shape, dtype=self.loc.dtype, device=self.loc.device)
        eps_D = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return (self.loc + _batch_mv(self._unbroadcasted_cov_factor, eps_W)
                + self._unbroadcasted_cov_diag.sqrt() * eps_D)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_lowrank_mahalanobis(self._unbroadcasted_cov_factor,
                                       self._unbroadcasted_cov_diag,
                                       diff,
                                       self._capacitance_tril)
        log_det = _batch_lowrank_logdet(self._unbroadcasted_cov_factor,
                                        self._unbroadcasted_cov_diag,
                                        self._capacitance_tril)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + log_det + M)


class MultivariateNormal(torch.distributions.MultivariateNormal, TorchDistributionMixin):
    support = IndependentConstraint(constraints.real, 1)  # TODO move upstream

    # Resolve the issue https://github.com/uber/pyro/issues/1586
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if 'covariance_matrix' in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(cov_shape)
        if 'scale_tril' in self.__dict__:
            new.scale_tril = self.scale_tril.expand(cov_shape)
        if 'precision_matrix' in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(cov_shape)
        super(torch.distributions.MultivariateNormal, new).__init__(batch_shape,
                                                                    self.event_shape,
                                                                    validate_args=False)
        new._validate_args = self._validate_args
        return new


class Independent(torch.distributions.Independent, TorchDistributionMixin):
    @constraints.dependent_property
    def support(self):
        return IndependentConstraint(self.base_dist.support, self.reinterpreted_batch_ndims)

    @property
    def _validate_args(self):
        return self.base_dist._validate_args

    @_validate_args.setter
    def _validate_args(self, value):
        self.base_dist._validate_args = value


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    kl = kl_divergence(p.base_dist, q.base_dist)
    if p.reinterpreted_batch_ndims:
        kl = sum_rightmost(kl, p.reinterpreted_batch_ndims)
    return kl


# Programmatically load all distributions from PyTorch.
__all__ = []
for _name, _Dist in torch.distributions.__dict__.items():
    if not isinstance(_Dist, type):
        continue
    if not issubclass(_Dist, torch.distributions.Distribution):
        continue
    if _Dist is torch.distributions.Distribution:
        continue

    try:
        _PyroDist = locals()[_name]
    except KeyError:
        _PyroDist = type(_name, (_Dist, TorchDistributionMixin), {})
        _PyroDist.__module__ = __name__
        locals()[_name] = _PyroDist

    _PyroDist.__doc__ = '''
    Wraps :class:`{}.{}` with
    :class:`~pyro.distributions.torch_distribution.TorchDistributionMixin`.
    '''.format(_Dist.__module__, _Dist.__name__)

    __all__.append(_name)


# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: pyro.distributions.{0}
    '''.format(_name)
    for _name in sorted(__all__)
])
