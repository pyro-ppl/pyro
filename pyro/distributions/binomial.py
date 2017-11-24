from __future__ import absolute_import, division, print_function

import torch
from pyro.distributions.multinomial import Multinomial


class Binomial(Multinomial):
    """
    Binomial distribution.

    Distribution over counts for `n` independent `Bernoulli(ps)` trials.

    :param torch.autograd.Variable ps: Probabilities. Should lie in the
        interval `[0,1]`.
    :param int n: Number of trials. Should be positive.
    """

    def __init__(self, ps, n, batch_size=None, *args, **kwargs):
        if ps.size(-1) != 1:
            raise ValueError("Parameter `ps` must have size 1 in the last dimension.")
        ps = torch.cat((1 - ps, ps), dim=-1)
        super(Binomial, self).__init__(ps, n, batch_size, *args, **kwargs)

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        counts = torch.sum(1 - self.expanded_sample(), dim=-1, keepdim=True)
        return counts

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.n * self.ps.index_select(-1, torch.LongTensor([1]))

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        ps = self.ps.index_select(-1, torch.LongTensor([1]))
        return self.n * ps * (1 - ps)
