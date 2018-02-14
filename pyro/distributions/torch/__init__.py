from __future__ import absolute_import, division, print_function

import numbers

import torch
from torch.autograd import Variable

from pyro.distributions.torch_distribution import TorchDistributionMixin


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
        if isinstance(n, Variable):
            n = n.data
        if not isinstance(n, numbers.Number):
            if n.max() != n.min():
                raise NotImplementedError('inhomogeneous n is not supported')
            n = n.view(-1)[0]
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


class TransformedDistribution(torch.distributions.TransformedDistribution, TorchDistributionMixin):
    pass


class Uniform(torch.distributions.Uniform, TorchDistributionMixin):
    def __init__(self, a, b):
        super(Uniform, self).__init__(a, b)
