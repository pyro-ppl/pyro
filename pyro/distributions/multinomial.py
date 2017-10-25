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
    def _sanitize_input(self, ps, n):
        if ps is not None:
            # stateless distribution
            return ps, n
        elif self.ps is not None:
            # stateful distribution
            return self.ps, self.n
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, ps=None, n=None, batch_size=None, *args, **kwargs):
        """
        Params:
          ps - probabilities
          n - num trials
        """
        self.ps = ps
        self.n = n
        if ps is not None:
            if ps.dim() == 1 and batch_size is not None:
                self.ps = ps.expand(batch_size, ps.size(0))
                self.n = n.expand(batch_size, n.size(0))
        super(Multinomial, self).__init__(*args, **kwargs)

    def batch_shape(self, ps=None, n=None):
        ps, n = self._sanitize_input(ps, n)
        return ps.size()[:-1]

    def event_shape(self, ps=None, n=None):
        ps, n = self._sanitize_input(ps, n)
        return ps.size()[-1:]

    def sample(self, ps=None, n=None):
        ps, n = self._sanitize_input(ps, n)
        counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=ps.size()[-1]),
                                     axis=-1,
                                     arr=self.expanded_sample(ps, n).data.cpu().numpy())
        counts = torch.from_numpy(counts)
        if ps.is_cuda:
            counts = counts.cuda()
        return Variable(counts)

    def expanded_sample(self, ps=None, n=None):
        ps, n = self._sanitize_input(ps, n)
        # get the int from Variable or Tensor
        if n.data.dim() == 2:
            n = int(n.data.cpu()[0][0])
        else:
            n = int(n.data.cpu()[0])
        return Variable(torch_multinomial(ps.data, n, replacement=True))

    def batch_log_pdf(self, x, ps=None, n=None):
        ps, n = self._sanitize_input(ps, n)
        log_factorial_n = log_gamma(x.sum(-1) + 1)
        log_factorial_xs = log_gamma(x + 1).sum(-1)
        log_powers = (x * torch.log(ps)).sum(-1)
        return log_factorial_n - log_factorial_xs + log_powers

    def analytic_mean(self, ps=None, n=None):
        ps, n = self._sanitize_input(ps, n)
        return n * ps

    def analytic_var(self, ps=None, n=None):
        ps, n = self._sanitize_input(ps, n)
        return n * ps * (1 - ps)
