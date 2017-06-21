import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


class Exponential(Distribution):
    """
    univariate exponential parameterized by lam
    """

    def __init__(self, lam, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        if lam.dim() == 1 and batch_size > 1:
            self.lam = lam.unsqueeze(0).expand(batch_size, lam.size(0))
        else:
            self.lam = lam
        self.reparametrized = True
        super(Exponential, self).__init__(*args, **kwargs)

    def sample(self):
        """
        reparameterized sampler.
        """
        eps = Variable(
            torch.rand(
                self.lam.size()),
            requires_grad=False).type_as(
            self.lam)
        x = -torch.log(eps) / self.lam
        return x

    def log_pdf(self, x):
        """
        exponential log-likelihood
        """
        ll = -self.lam * x + torch.log(self.lam)
        return torch.sum(ll)

    def batch_log_pdf(self, x, batch_size=1):
        """
        exponential log-likelihood
        """
        if x.dim() == 1 and self.lam.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll = -self.lam * x + torch.log(self.lam)
        return torch.sum(ll, 1)
