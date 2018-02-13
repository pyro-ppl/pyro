from __future__ import absolute_import, division, print_function

import numbers

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Bernoulli(Distribution, torch.distributions.Bernoulli):
    def __init__(self, ps=None, logits=None):
        super(Bernoulli, self).__init__(probs=ps, logits=logits)


class Beta(Distribution, torch.distributions.Beta):
    def __init__(self, alpha, beta):
        super(Beta, self).__init__(alpha, beta)


class Binomial(Distribution, torch.distributions.Binomial):
    def __init__(self, n, ps):
        super(Binomial, self).__init__(n, ps)


class Categorical(Distribution, torch.distributions.Categorical):
    def __init__(self, ps=None, logits=None):
        super(Categorical, self).__init__(probs=ps, logits=logits)


class Cauchy(Distribution, torch.distributions.Cauchy):
    def __init__(self, mu, gamma):
        super(Cauchy, self).__init__(mu, gamma)


class Dirichlet(Distribution, torch.distributions.Dirichlet):
    def __init__(self, alpha):
        super(Dirichlet, self).__init__(alpha)


class Exponential(Distribution, torch.distributions.Exponential):
    def __init__(self, lam):
        super(Exponential, self).__init__(lam)


class Gamma(Distribution, torch.distributions.Gamma):
    def __init__(self, alpha, beta):
        super(Gamma, self).__init__(alpha, beta)


class LogNormal(Distribution, torch.distributions.LogNormal):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        super(LogNormal, self).__init__(mu, sigma)


class Multinomial(Distribution, torch.distributions.Multinomial):
    def __init__(self, ps, n=1):
        if isinstance(n, Variable):
            n = n.data
        if not isinstance(n, numbers.Number):
            if n.max() != n.min():
                raise NotImplementedError('inhomogeneous n is not supported')
            n = n.view(-1)[0]
        n = int(n)
        super(Multinomial, self).__init__(n, probs=ps)


class Normal(Distribution, torch.distributions.Normal):
    def __init__(self, mu, sigma):
        super(Normal, self).__init__(mu, sigma)


class OneHotCategorical(Distribution, torch.distributions.OneHotCategorical):
    def __init__(self, ps=None, logits=None):
        super(OneHotCategorical, self).__init__(probs=ps, logits=logits)


class Poisson(Distribution, torch.distributions.Poisson):
    def __init__(self, lam):
        super(Poisson, self).__init__(lam)


class TransformedDistribution(Distribution, torch.distributions.TransformedDistribution):
    pass


class Uniform(Distribution, torch.distributions.Uniform):
    def __init__(self, a, b):
        super(Uniform, self).__init__(a, b)
