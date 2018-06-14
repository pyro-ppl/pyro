from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.distributions.util import ShapeMismatchError


class Bernoulli(torch.distributions.Bernoulli, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape)
                return Bernoulli(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape)
                return Bernoulli(logits=logits, validate_args=validate_args)


class Beta(torch.distributions.Beta, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            concentration1 = self.concentration1.expand(batch_shape)
            concentration0 = self.concentration0.expand(batch_shape)
            beta = Beta(concentration1, concentration0, validate_args=validate_args)
            beta.has_rsample = getattr(self, 'has_rsample')
            return beta


class Categorical(torch.distributions.Categorical, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape + self.probs.shape[-1:])
                return Categorical(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape + self.logits.shape[-1:])
                return Categorical(logits=logits, validate_args=validate_args)


class Cauchy(torch.distributions.Cauchy, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            cauchy = Cauchy(loc, scale, validate_args=validate_args)
            cauchy.has_rsample = getattr(self, 'has_rsample')
            return cauchy


class Chi2(torch.distributions.Chi2, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            df = self.df.expand(batch_shape)
            chi2 = Chi2(df, validate_args=validate_args)
            chi2.has_rsample = getattr(self, 'has_rsample')
            return chi2


class Dirichlet(torch.distributions.Dirichlet, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('validate_args')
            concentration = self.concentration.expand(batch_shape + self.event_shape)
            dirichlet = Dirichlet(concentration, validate_args=validate_args)
            dirichlet.has_rsample = getattr(self, 'has_rsample')
            return dirichlet


class Exponential(torch.distributions.Exponential, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            rate = self.rate.expand(batch_shape)
            exponential = Exponential(rate, validate_args=validate_args)
            exponential.has_rsample = getattr(self, 'has_rsample')
            return exponential


class Gamma(torch.distributions.Gamma, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            concentration = self.concentration.expand(batch_shape)
            rate = self.rate.expand(batch_shape)
            gamma = Gamma(concentration, rate, validate_args=validate_args)
            gamma.has_rsample = getattr(self, 'has_rsample')
            return gamma


class Geometric(torch.distributions.Geometric, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape)
                return Geometric(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape)
                return Geometric(logits=logits, validate_args=validate_args)


class Gumbel(torch.distributions.Gumbel, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return Gumbel(loc, scale, validate_args=validate_args)


class Independent(torch.distributions.Independent, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('validate_args')
            extra_shape = self.base_dist.event_shape[:self.reinterpreted_batch_ndims]
            base_dist = self.base_dist.expand(batch_shape + extra_shape)
            return Independent(base_dist, self.reinterpreted_batch_ndims, validate_args=validate_args)


class Laplace(torch.distributions.Laplace, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            laplace = Laplace(loc, scale, validate_args=validate_args)
            laplace.has_rsample = getattr(self, 'has_rsample')
            return laplace


class LogNormal(torch.distributions.LogNormal, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            return LogNormal(loc, scale, validate_args=validate_args)


class Multinomial(torch.distributions.Multinomial, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape + self.event_shape)
                return Multinomial(self.total_count, probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape + self.event_shape)
                return Multinomial(self.total_count, logits=logits, validate_args=validate_args)


class MultivariateNormal(torch.distributions.MultivariateNormal, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('validate_args')
            loc = self.loc.expand(batch_shape + self.event_shape)
            if 'scale_tril' in self.__dict__:
                scale_tril = self.scale_tril.expand(batch_shape + self.event_shape + self.event_shape)
                mvn = MultivariateNormal(loc, scale_tril=scale_tril, validate_args=validate_args)
            elif 'covariance_matrix' in self.__dict__:
                covariance_matrix = self.covariance_matrix.expand(batch_shape + self.event_shape + self.event_shape)
                mvn = MultivariateNormal(loc, covariance_matrix=covariance_matrix, validate_args=validate_args)
            else:
                precision_matrix = self.precision_matrix.expand(batch_shape + self.event_shape + self.event_shape)
                mvn = MultivariateNormal(loc, precision_matrix=precision_matrix, validate_args=validate_args)
            mvn.has_rsample = getattr(self, 'has_rsample')
            return mvn


class Normal(torch.distributions.Normal, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            normal = Normal(loc, scale, validate_args=validate_args)
            normal.has_rsample = getattr(self, 'has_rsample')
            return normal


class OneHotCategorical(torch.distributions.OneHotCategorical, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            batch_shape = torch.Size(batch_shape)
            validate_args = self.__dict__.get('validate_args')
            if 'probs' in self.__dict__:
                probs = self.probs.expand(batch_shape + self.event_shape)
                return OneHotCategorical(probs=probs, validate_args=validate_args)
            else:
                logits = self.logits.expand(batch_shape + self.event_shape)
                return OneHotCategorical(logits=logits, validate_args=validate_args)


class Poisson(torch.distributions.Poisson, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            rate = self.rate.expand(batch_shape)
            return Poisson(rate, validate_args=validate_args)


class StudentT(torch.distributions.StudentT, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            df = self.df.expand(batch_shape)
            loc = self.loc.expand(batch_shape)
            scale = self.scale.expand(batch_shape)
            student_t = StudentT(df, loc, scale, validate_args=validate_args)
            student_t.has_rsample = getattr(self, 'has_rsample')
            return student_t


class TransformedDistribution(torch.distributions.TransformedDistribution, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            base_dist = self.base_dist.expand(batch_shape)
            return TransformedDistribution(base_dist, self.transforms)


class Uniform(torch.distributions.Uniform, TorchDistributionMixin):
    def expand(self, batch_shape):
        try:
            sample_shape = self._sample_shape(batch_shape)
            return self.expand_by(sample_shape)
        except ShapeMismatchError:
            validate_args = self.__dict__.get('validate_args')
            low = self.low.expand(batch_shape)
            high = self.high.expand(batch_shape)
            uniform = Uniform(low, high, validate_args=validate_args)
            uniform.has_rsample = getattr(self, 'has_rsample')
            return uniform


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
