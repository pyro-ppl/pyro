import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Exponential(Distribution):
    """
    :param lam: rate *(real (0, Infinity))*

    Exponential parameterized by lambda
    """
    reparameterized = True

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
        Params:
          `lam` - lambda
        """
        self.lam = lam
        if lam is not None:
            if lam.dim() == 1 and batch_size is not None:
                self.lam = lam.expand(batch_size, lam.size(0))
        super(Exponential, self).__init__(*args, **kwargs)

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
        reparameterized sampler.
        """
        lam = self._sanitize_input(lam)
        eps = Variable(torch.rand(lam.size()).type_as(lam.data))
        x = -torch.log(eps) / lam
        return x

    def batch_log_pdf(self, x, lam=None, batch_size=1):
        """
        exponential log-likelihood
        """
        lam = self._sanitize_input(lam)
        if lam.size() != x.size():
            lam = lam.expand_as(x)
        ll = - lam * x + torch.log(lam)
        batch_log_pdf_shape = self.batch_shape(lam) + (1,)
        return torch.sum(ll, -1).contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self, lam=None):
        lam = self._sanitize_input(lam)
        return torch.pow(lam, -1.0)

    def analytic_var(self, lam=None):
        lam = self._sanitize_input(lam)
        return torch.pow(lam, -2.0)
