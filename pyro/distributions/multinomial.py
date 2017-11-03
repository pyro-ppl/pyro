from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma, torch_multinomial


class Multinomial(Distribution):
    """
    Multinomial distribution.

    Distribution over counts for `n` independent `Categorical(ps)` trials.

    This is often used in conjunction with `torch.nn.Softmax` to ensure
    probabilites `ps` are normalized.

    :param torch.autograd.Variable ps: Probabilities (real). Should be positive
        and should normalized over the rightmost axis.
    :param int n: Number of trials. Should be positive.
    """

    def __init__(self, ps, n, batch_size=None, *args, **kwargs):
        if ps.dim() not in (1, 2):
            raise ValueError("Parameter `ps` must be either 1 or 2 dimensional.")
        self.ps = ps
        self.n = n
        if ps.dim() == 1 and batch_size is not None:
            self.ps = ps.expand(batch_size, ps.size(0))
            self.n = n.expand(batch_size, n.size(0))
        super(Multinomial, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        ps = self.ps
        if x is not None:
            if x.size()[-event_dim] != ps.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.ps.size()[-1], but got {} vs {}".format(
                                     x.size(-1), ps.size(-1)))
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
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self.ps.size()[-1]),
            axis=-1,
            arr=self.expanded_sample().data.cpu().numpy())
        counts = torch.from_numpy(counts)
        if self.ps.is_cuda:
            counts = counts.cuda()
        return Variable(counts)

    def expanded_sample(self):
        # get the int from Variable or Tensor
        if self.n.data.dim() == 2:
            n = int(self.n.data.cpu()[0][0])
        else:
            n = int(self.n.data.cpu()[0])
        return Variable(torch_multinomial(self.ps.data, n, replacement=True))

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_factorial_n = log_gamma(x.sum(-1) + 1)
        log_factorial_xs = log_gamma(x + 1).sum(-1)
        log_powers = (x * torch.log(self.ps)).sum(-1)
        batch_log_pdf = log_factorial_n - log_factorial_xs + log_powers
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
