import numpy.random as npr
import torch
from torch.autograd import Variable
import pyro
from pyro.distributions.distribution import Distribution
from pyro.util import log_gamma


class Poisson(Distribution):
    """
    :param lam: mean *(real (0, Infinity))*

    Poisson distribution over integers parameterizeds by lambda.
    """

    def _sanitize_input(self, lam):
        if lam is not None:
            # stateless distribution
            return lam
        elif self.lam is not None:
            # stateful distribution
            return self.lam
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, lam=None, batch_size=1, *args, **kwargs):
        """
          `lam` - rate parameter
        """
        self.lam = lam
        if lam is not None:
            if lam.dim() == 1 and batch_size > 1:
                self.lam = lam.unsqueeze(0).expand(batch_size, lam.size(0))
        super(Poisson, self).__init__(*args, **kwargs)

    def sample(self, lam=None, *args, **kwargs):
        """
        Poisson sampler.
        """
        _lam = self._sanitize_input(lam)
        x = npr.poisson(lam=_lam.data.numpy()).astype("float")
        return Variable(torch.Tensor(x).type_as(_lam.data))

    def log_pdf(self, x, lam=None, *args, **kwargs):
        """
        Poisson log-likelihood
        Warning: need pytorch implementation of log gamma in order to be differentiable
        """
        _lam = self._sanitize_input(lam)
        ll_1 = torch.sum(x * torch.log(_lam))
        ll_2 = -torch.sum(_lam)
        ll_3 = -torch.sum(log_gamma(x + 1.0))
        return ll_1 + ll_2 + ll_3

    def batch_log_pdf(self, x, lam=None, batch_size=1, *args, **kwargs):
        _lam = self._sanitize_input(lam)
        if x.dim() == 1 and _lam.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, _lam)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = torch.sum(x * torch.log(_lam), 1)
        ll_2 = -torch.sum(_lam, 1)
        ll_3 = -torch.sum(log_gamma(x + 1.0), 1)
        return ll_1 + ll_2 + ll_3
