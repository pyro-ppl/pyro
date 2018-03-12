from __future__ import absolute_import, division, print_function

import numbers

import torch

from pyro.distributions.torch_distribution import TorchDistributionMixin

# These distributions require custom wrapping.
# TODO rename parameters so these can be imported automatically.


class Bernoulli(torch.distributions.Bernoulli, TorchDistributionMixin):
    def __init__(self, ps=None, logits=None):
        super(Bernoulli, self).__init__(probs=ps, logits=logits)


class Beta(torch.distributions.Beta, TorchDistributionMixin):
    def __init__(self, alpha, beta):
        super(Beta, self).__init__(alpha, beta)


class Binomial(torch.distributions.Binomial, TorchDistributionMixin):
    def __init__(self, n, ps):
        super(Binomial, self).__init__(n, ps)


class Categorical(torch.distributions.Categorical, TorchDistributionMixin):
    def __init__(self, ps=None, logits=None):
        super(Categorical, self).__init__(probs=ps, logits=logits)


class Cauchy(torch.distributions.Cauchy, TorchDistributionMixin):
    def __init__(self, mu, gamma):
        super(Cauchy, self).__init__(mu, gamma)


class Dirichlet(torch.distributions.Dirichlet, TorchDistributionMixin):
    def __init__(self, alpha):
        super(Dirichlet, self).__init__(alpha)


class Exponential(torch.distributions.Exponential, TorchDistributionMixin):
    def __init__(self, lam):
        super(Exponential, self).__init__(lam)


class Gamma(torch.distributions.Gamma, TorchDistributionMixin):
    def __init__(self, alpha, beta):
        super(Gamma, self).__init__(alpha, beta)


class LogNormal(torch.distributions.LogNormal, TorchDistributionMixin):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        super(LogNormal, self).__init__(mu, sigma)


class Multinomial(torch.distributions.Multinomial, TorchDistributionMixin):
    def __init__(self, ps, n=1):
        if not isinstance(n, numbers.Number):
            if n.max() != n.min():
                raise NotImplementedError('inhomogeneous n is not supported')
            n = n.data.view(-1)[0]
        n = int(n)
        super(Multinomial, self).__init__(n, probs=ps)


class Normal(torch.distributions.Normal, TorchDistributionMixin):
    def __init__(self, mu, sigma):
        super(Normal, self).__init__(mu, sigma)


class OneHotCategorical(torch.distributions.OneHotCategorical, TorchDistributionMixin):
    def __init__(self, ps=None, logits=None):
        super(OneHotCategorical, self).__init__(probs=ps, logits=logits)


class Poisson(torch.distributions.Poisson, TorchDistributionMixin):
    def __init__(self, lam):
        super(Poisson, self).__init__(lam)


class Uniform(torch.distributions.Uniform, TorchDistributionMixin):
    def __init__(self, a, b):
        super(Uniform, self).__init__(a, b)


# Programmatically load all remaining distributions.
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
    .. autoclass:: {0}
    '''.format(_name)
    for _name in sorted(__all__)
])
