from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints

from pyro.distributions.torch_distribution import IndependentConstraint, TorchDistributionMixin


class Bernoulli(torch.distributions.Bernoulli, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Bernoulli, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape)
                return type(self)(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape)
                return type(self)(logits=logits, validate_args=validate_args)

    def enumerate_support(self, expand=True):
        values = self._param.new_tensor([0., 1.])
        values = values.reshape((2,) + (1,) * len(self.batch_shape))
        if expand:
            values = values.expand((2,) + self.batch_shape)
        return values


class Beta(torch.distributions.Beta, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Beta, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            concentration1 = self.concentration1.expand(batch_shape)
            concentration0 = self.concentration0.expand(batch_shape)
            return type(self)(concentration1, concentration0, validate_args=validate_args)


class Categorical(torch.distributions.Categorical, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Categorical, self).expand(batch_shape)
        except NotImplementedError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('_validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape + self.probs.shape[-1:])
                return type(self)(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape + self.logits.shape[-1:])
                return type(self)(logits=logits, validate_args=validate_args)

    def enumerate_support(self, expand=True):
        num_events = self._num_events
        values = torch.arange(num_events, dtype=torch.long)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        if self._param.is_cuda:
            values = values.cuda(self._param.get_device())
        return values


class Cauchy(torch.distributions.Cauchy, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Cauchy, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return type(self)(loc, scale, validate_args=validate_args)


class Chi2(torch.distributions.Chi2, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Chi2, self).expand_by(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            df = self.df.expand(batch_shape)
            return type(self)(df, validate_args=validate_args)


class Dirichlet(torch.distributions.Dirichlet, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Dirichlet, self).expand(batch_shape)
        except NotImplementedError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('_validate_args')
            concentration = self.concentration.expand(batch_shape + self.event_shape)
            return type(self)(concentration, validate_args=validate_args)


class Exponential(torch.distributions.Exponential, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Exponential, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            rate = self.rate.expand(batch_shape)
            return type(self)(rate, validate_args=validate_args)


class Gamma(torch.distributions.Gamma, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Gamma, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            concentration = self.concentration.expand(batch_shape)
            rate = self.rate.expand(batch_shape)
            return type(self)(concentration, rate, validate_args=validate_args)


class Geometric(torch.distributions.Geometric, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Geometric, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape)
                return type(self)(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape)
                return type(self)(logits=logits, validate_args=validate_args)


class Gumbel(torch.distributions.Gumbel, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Gumbel, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return type(self)(loc, scale, validate_args=validate_args)


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

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        base_shape = self.base_dist.batch_shape
        reinterpreted_shape = base_shape[len(base_shape) - self.reinterpreted_batch_ndims:]
        base_dist = self.base_dist.expand(batch_shape + reinterpreted_shape)
        return type(self)(base_dist, self.reinterpreted_batch_ndims)

    def enumerate_support(self, expand=expand):
        if self.reinterpreted_batch_ndims:
            raise NotImplementedError("Pyro does not enumerate over cartesian products")
        return self.base_dist.enumerate_support(expand=expand)


class Laplace(torch.distributions.Laplace, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Laplace, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return type(self)(loc, scale, validate_args=validate_args)


class LogNormal(torch.distributions.LogNormal, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(LogNormal, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return type(self)(loc, scale, validate_args=validate_args)


class Multinomial(torch.distributions.Multinomial, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Multinomial, self).expand(batch_shape)
        except NotImplementedError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('_validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape + self.event_shape)
                return type(self)(self.total_count, probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape + self.event_shape)
                return type(self)(self.total_count, logits=logits, validate_args=validate_args)


class MultivariateNormal(torch.distributions.MultivariateNormal, TorchDistributionMixin):
    support = IndependentConstraint(constraints.real, 1)  # TODO move upstream

    def expand(self, batch_shape):
        try:
            return super(MultivariateNormal, self).expand(batch_shape)
        except NotImplementedError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape + self.event_shape)
            if 'scale_tril' in self.__dict__:
                scale_tril = self.scale_tril.expand(batch_shape + self.event_shape + self.event_shape)
                return type(self)(loc, scale_tril=scale_tril, validate_args=validate_args)
            elif 'covariance_matrix' in self.__dict__:
                covariance_matrix = self.covariance_matrix.expand(batch_shape + self.event_shape + self.event_shape)
                return type(self)(loc, covariance_matrix=covariance_matrix, validate_args=validate_args)
            else:
                precision_matrix = self.precision_matrix.expand(batch_shape + self.event_shape + self.event_shape)
                return type(self)(loc, precision_matrix=precision_matrix, validate_args=validate_args)


class Normal(torch.distributions.Normal, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Normal, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return type(self)(loc, scale, validate_args=validate_args)


class OneHotCategorical(torch.distributions.OneHotCategorical, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(OneHotCategorical, self).expand(batch_shape)
        except NotImplementedError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('_validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape + self.event_shape)
                return type(self)(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape + self.event_shape)
                return type(self)(logits=logits, validate_args=validate_args)

    def enumerate_support(self, expand=True):
        n = self.event_shape[0]
        values = self._new((n, n))
        torch.eye(n, out=values)
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        if expand:
            values = values.expand((n,) + self.batch_shape + (n,))
        return values


class Poisson(torch.distributions.Poisson, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Poisson, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            rate = self.rate.expand(batch_shape)
            return type(self)(rate, validate_args=validate_args)


class StudentT(torch.distributions.StudentT, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(StudentT, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            df = self.df.expand(batch_shape)
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return type(self)(df, loc, scale, validate_args=validate_args)


class TransformedDistribution(torch.distributions.TransformedDistribution, TorchDistributionMixin):
    def expand(self, batch_shape):
        return super(TransformedDistribution, self).expand(batch_shape)


class Uniform(torch.distributions.Uniform, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            return super(Uniform, self).expand(batch_shape)
        except NotImplementedError:
            validate_args = self.__dict__.get('_validate_args')
            low = self.low.expand(batch_shape)
            high = self.high.expand(batch_shape)
            return type(self)(low, high, validate_args=validate_args)


# Programmatically load all distributions from PyTorch.
__all__ = []
for _name, _Dist in torch.distributions.__dict__.items():
    if _name == 'Binomial':
        continue
    if not isinstance(_Dist, type):
        continue
    if not issubclass(_Dist, torch.distributions.Distribution):
        continue
    if _Dist is torch.distributions.Distribution:
        continue

    try:
        _PyroDist = locals()[_name]
    except KeyError:

        class _PyroDist(_Dist, TorchDistributionMixin):
            pass

        _PyroDist.__name__ = _name
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
