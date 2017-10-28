import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma, torch_multinomial


class Multinomial(Distribution):
    """
    :param ps: probabilities *(real array with elements that sum to one)*
    :param n: number of trials *(int (>=1))*

    Distribution over counts for ``n`` independent ``Discrete({ps: ps})``
    trials.
    """

    def __init__(self, ps, n, batch_size=None, *args, **kwargs):
        """
        Params:
          ps - probabilities
          n - num trials
        """
        if ps.dim() not in (1, 2):
            raise ValueError("Parameter `ps` must be either 1 or 2 dimensional.")
        self.ps = ps
        self.n = n
        if ps.dim() == 1 and batch_size is not None:
            self.ps = ps.expand(batch_size, ps.size(0))
            self.n = n.expand(batch_size, n.size(0))
        super(Multinomial, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        event_dim = 1
        ps = self.ps
        if x is not None and x.size() != ps.size():
            ps = self.ps.expand(x.size()[:-event_dim] + self.event_shape())
        return ps.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.ps.size()[-1]),
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
        log_factorial_n = log_gamma(x.sum(-1) + 1)
        log_factorial_xs = log_gamma(x + 1).sum(-1)
        log_powers = (x * torch.log(self.ps)).sum(-1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        batch_log_pdf = log_factorial_n - log_factorial_xs + log_powers
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        return self.n * self.ps

    def analytic_var(self):
        return self.n * self.ps * (1 - self.ps)
