from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints, kl_divergence, register_kl

from pyro.distributions.torch_distribution import IndependentConstraint, TorchDistributionMixin
from pyro.distributions.util import sum_rightmost


class Bernoulli(torch.distributions.Bernoulli, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Bernoulli, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Bernoulli, self).expand(batch_shape, _instance)


class Beta(torch.distributions.Beta, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Beta, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Beta, self).expand(batch_shape, _instance)


class Categorical(torch.distributions.Categorical, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Categorical, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Categorical, self).expand(batch_shape, _instance)


class Cauchy(torch.distributions.Cauchy, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Cauchy, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Cauchy, self).expand(batch_shape, _instance)


class Chi2(torch.distributions.Chi2, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Chi2, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Chi2, self).expand(batch_shape, _instance)


class Dirichlet(torch.distributions.Dirichlet, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Dirichlet, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Dirichlet, self).expand(batch_shape, _instance)


class Exponential(torch.distributions.Exponential, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Exponential, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Exponential, self).expand(batch_shape, _instance)


class Gamma(torch.distributions.Gamma, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Gamma, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Gamma, self).expand(batch_shape, _instance)


class Geometric(torch.distributions.Geometric, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Geometric, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Geometric, self).expand(batch_shape, _instance)


class Gumbel(torch.distributions.Gumbel, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Gumbel, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Gumbel, self).expand(batch_shape, _instance)


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


class Laplace(torch.distributions.Laplace, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Laplace, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Laplace, self).expand(batch_shape, _instance)


class LogNormal(torch.distributions.LogNormal, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(LogNormal, self)._expand(batch_shape)
        except NotImplementedError:
            return super(LogNormal, self).expand(batch_shape, _instance)


class Multinomial(torch.distributions.Multinomial, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Multinomial, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Multinomial, self).expand(batch_shape, _instance)


class MultivariateNormal(torch.distributions.MultivariateNormal, TorchDistributionMixin):
    support = IndependentConstraint(constraints.real, 1)  # TODO move upstream

    def expand(self, batch_shape, _instance=None):
        try:
            return super(MultivariateNormal, self)._expand(batch_shape)
        except NotImplementedError:
            return super(MultivariateNormal, self).expand(batch_shape, _instance)


class Normal(torch.distributions.Normal, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Normal, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Normal, self).expand(batch_shape, _instance)


class OneHotCategorical(torch.distributions.OneHotCategorical, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(OneHotCategorical, self)._expand(batch_shape)
        except NotImplementedError:
            return super(OneHotCategorical, self).expand(batch_shape, _instance)


class Poisson(torch.distributions.Poisson, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Poisson, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Poisson, self).expand(batch_shape, _instance)


class StudentT(torch.distributions.StudentT, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(StudentT, self)._expand(batch_shape)
        except NotImplementedError:
            return super(StudentT, self).expand(batch_shape, _instance)


class TransformedDistribution(torch.distributions.TransformedDistribution, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(TransformedDistribution, self)._expand(batch_shape)
        except NotImplementedError:
            return super(TransformedDistribution, self).expand(batch_shape)


class Uniform(torch.distributions.Uniform, TorchDistributionMixin):
    def expand(self, batch_shape, _instance=None):
        try:
            return super(Uniform, self)._expand(batch_shape)
        except NotImplementedError:
            return super(Uniform, self).expand(batch_shape, _instance)


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
