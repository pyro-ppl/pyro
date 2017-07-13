import scipy.stats as spr
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution


def log_gamma(xx):
    """
    quick and dirty log gamma copied from webppl
    """
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = Variable(torch.ones(x.size())) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


class Beta(Distribution):
    """
    univariate beta distribution parameterized by alpha and beta
    """

    def __init__(self, alpha, beta, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        if alpha.dim() != beta.dim():
            raise ValueError("Alpha and beta need to have the same dimensions.")
        if alpha.dim() == 1 and beta.dim() == 1:
            self.alpha = alpha.expand(batch_size, 0)
            self.beta = beta.expand(batch_size, 0)
        else:
            self.alpha = alpha
            self.beta = beta
        self.reparametrized = False
        super(Beta, self).__init__(*args, **kwargs)

    def sample(self):
        """
        un-reparameterizeable sampler.
        """
        x = Variable(torch.Tensor(
            [spr.beta.rvs(self.alpha.data.cpu().numpy(), self.beta.data.cpu().numpy())]))
        return x

    def log_pdf(self, x):
        """
        gamma log-likelihood
        """
        one = Variable(torch.ones(self.alpha.size()))
        ll_1 = (self.alpha - one) * torch.log(x)
        ll_2 = (self.beta - one) * torch.log(one - x)
        ll_3 = log_gamma(self.alpha + self.beta)
        ll_4 = -log_gamma(self.alpha)
        ll_5 = -log_gamma(self.beta)
        return ll_1 + ll_2 + ll_3 + ll_4 + ll_5

    def batch_log_pdf(self, x, batch_size=1):
        if x.dim() == 1 and self.beta.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        one = Variable(torch.ones(x.size()))
        ll_1 = (self.alpha - one) * torch.log(x)
        ll_2 = (self.beta - one) * torch.log(one - x)
        ll_3 = log_gamma(self.alpha + self.beta)
        ll_4 = -log_gamma(self.alpha)
        ll_5 = -log_gamma(self.beta)
        return ll_1 + ll_2 + ll_3 + ll_4 + ll_5
