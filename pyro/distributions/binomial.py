from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma, torch_multinomial


class Binomial(Distribution):
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
        if ps.dim() not in (1, 2):
            raise ValueError("Parameter `ps` must be either 1 or 2 dimensional.")
        self.ps = ps
        if ps.dim() == 1 and batch_size is not None:
            self.ps = ps.expand(batch_size, ps.size(0))
        if isinstance(n, int):
            n = torch.LongTensor([n])
            if self.ps.is_cuda:
                n = n.cuda()
            n = Variable(n)
        self.n = n.expand_as(self.ps)
        super(Binomial, self).__init__(*args, **kwargs)
        
    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        ps = self.ps
        if x is not None:
            if x.size()[-event_dim] != 1:
                raise ValueError("The event size for the data must be 1.\n"
                                 "Expected x.size()[-1] == 1, but got {}".format(x.size(-1)))
            try:
                ps = self.ps.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `ps` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(ps.size(), x.size(), str(e)))
        return ps.size()[:-event_dim]
        
    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        return 1

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        counts = torch.sum(1 - self.expanded_sample(), dim=-1, keepdim=True)
        return counts
    
    def expanded_sample(self):
        # get the int from Variable or Tensor
        if self.n.data.dim() == 2:
            n = int(self.n.data.cpu()[0][0])
        else:
            n = int(self.n.data.cpu()[0])
        ps = torch.cat((1 - self.ps, self.ps), dim=-1)
        return Variable(torch_multinomial(ps.data, n, replacement=True))
    
    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_factorial_n = log_gamma(self.n + 1)
        log_factorial_x = log_gamma(x + 1)
        log_powers = x * torch.log(self.ps) + (self.n - x) * torch.log(1 - self.ps)
        batch_log_pdf = log_factorial_n - log_factorial_x + log_powers
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.n * self.ps

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return self.n * self.ps * (1 - self.ps)
