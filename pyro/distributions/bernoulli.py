import functools
import itertools

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Bernoulli(Distribution):
    """
    :param ps: probabilities *(vector [0, 1])*

    Distribution over a vector of independent Bernoulli variables. Each element
    of the vector takes on a value in ``{0, 1}``.
    """
    enumerable = True

    def _sanitize_input(self, ps):
        if ps is not None:
            # stateless distribution
            return ps
        elif self.ps is not None:
            # stateful distribution
            return self.ps
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, ps=None, batch_size=1, *args, **kwargs):
        """
        Params:
          ps = tensor of probabilities
        """
        self.ps = ps
        if ps is not None:
            if ps.dim() == 1:
                self.ps = ps.expand(batch_size, ps.size(0))
        super(Bernoulli, self).__init__(*args, **kwargs)

    def sample(self, ps=None, *args, **kwargs):
        """
        Bernoulli sampler.
        """
        ps = self._sanitize_input(ps)
        return Variable(torch.bernoulli(ps.data).type_as(ps.data))

    def log_pdf(self, x, ps=None, batch_size=1, *args, **kwargs):
        """
        Bernoulli log-likelihood
        """
        ps = self._sanitize_input(ps)
        x_1 = x - 1
        ps_1 = ps - 1
        xmul = torch.mul(x, ps)
        xmul_1 = torch.mul(x_1, ps_1)
        logsum = torch.log(torch.add(xmul, xmul_1))
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if 'log_pdf_mask' in kwargs:
            return torch.sum(kwargs['log_pdf_mask'] * logsum)
        return torch.sum(logsum)

    def batch_log_pdf(self, x, ps=None, batch_size=1, *args, **kwargs):
        ps = self._sanitize_input(ps)
        if x.dim() == 1 and ps.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, ps)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        if ps.size() != x.size():
            ps = ps.expand_as(x)
        x_1 = x - 1
        ps_1 = ps - 1
        xmul = torch.mul(x, ps)
        xmul_1 = torch.mul(x_1, ps_1)
        logsum = torch.log(torch.add(xmul, xmul_1))
        return torch.sum(logsum, 1)

    def support(self, ps=None, *args, **kwargs):
        ps = self._sanitize_input(ps)
        if ps.dim() == 1:
            return iter([Variable(torch.ones(1).type_as(ps.data)), Variable(torch.zeros(1).type_as(ps))])
        size = functools.reduce(lambda x, y: x * y, ps.size())
        return (Variable(torch.Tensor(list(x)).view_as(ps))
                for x in itertools.product(torch.Tensor([0, 1]).type_as(ps.data), repeat=size))

    def analytic_mean(self, ps=None):
        ps = self._sanitize_input(ps)
        return ps

    def analytic_var(self, ps=None):
        ps = self._sanitize_input(ps)
        return ps * (1 - ps)
