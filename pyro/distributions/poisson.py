import numpy.random as npr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma


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

    def __init__(self, lam=None, batch_size=None, *args, **kwargs):
        """
          `lam` - rate parameter
        """
        self.lam = lam
        if lam is not None:
            if lam.dim() == 1 and batch_size is not None:
                self.lam = lam.expand(batch_size, lam.size(0))
        super(Poisson, self).__init__(*args, **kwargs)

    def batch_shape(self, lam=None):
        lam = self._sanitize_input(lam)
        event_dim = 1
        return lam.size()[:-event_dim]

    def event_shape(self, lam=None):
        lam = self._sanitize_input(lam)
        event_dim = 1
        return lam.size()[-event_dim:]

    def sample(self, lam=None):
        """
        Poisson sampler.
        """
        lam = self._sanitize_input(lam)
        x = npr.poisson(lam=lam.data.cpu().numpy()).astype("float")
        return Variable(torch.Tensor(x).type_as(lam.data))

    def batch_log_pdf(self, x, lam=None):
        """
        Poisson log-likelihood
        NOTE: Requires Pytorch implementation of log_gamma to be differentiable
        """
        lam = self._sanitize_input(lam)
        if lam.size() != x.size():
            lam = lam.expand_as(x)
        ll_1 = torch.sum(x * torch.log(lam), -1)
        ll_2 = -torch.sum(lam, -1)
        ll_3 = -torch.sum(log_gamma(x + 1.0), -1)
        batch_log_pdf = ll_1 + ll_2 + ll_3
        batch_log_pdf_shape = self.batch_shape(lam) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self, lam=None):
        lam = self._sanitize_input(lam)
        return lam

    def analytic_var(self, lam=None):
        lam = self._sanitize_input(lam)
        return lam
