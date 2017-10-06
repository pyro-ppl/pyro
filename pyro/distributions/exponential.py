import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Exponential(Distribution):
    """
    :param lam: rate *(real (0, Infinity))*

    Exponential parameterized by lambda
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
        Params:
          `lam` - lambda
        """
        self.lam = lam
        if lam is not None:
            if lam.dim() == 1 and batch_size > 1:
                self.lam = lam.expand(batch_size, lam.size(0))
        self.reparameterized = True
        super(Exponential, self).__init__(*args, **kwargs)

    def sample(self, lam=None, *args, **kwargs):
        """
        reparameterized sampler.
        """
        lam = self._sanitize_input(lam)
        eps = Variable(torch.rand(lam.size()).type_as(lam.data))
        x = -torch.log(eps) / lam
        return x

    def log_pdf(self, x, lam=None, *args, **kwargs):
        """
        exponential log-likelihood
        """
        lam = self._sanitize_input(lam)
        ll = - lam * x + torch.log(lam)
        return torch.sum(ll)

    def batch_log_pdf(self, x, lam=None, batch_size=1, *args, **kwargs):
        """
        exponential log-likelihood
        """
        lam = self._sanitize_input(lam)
        if x.dim() == 1 and lam.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, lam)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll = - lam * x + torch.log(lam)
        return torch.sum(ll, 1)

    def analytic_mean(self, lam=None):
        lam = self._sanitize_input(lam)
        return torch.pow(lam, -1.0)

    def analytic_var(self, lam=None):
        lam = self._sanitize_input(lam)
        return torch.pow(lam, -2.0)
