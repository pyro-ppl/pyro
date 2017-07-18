import scipy.stats as spr
import torch
from torch.autograd import Variable
import pyro
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
    ser = pyro.ones(x.size()) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


class Gamma(Distribution):
    """
    univariate gamma parameterized by alpha and beta
    """

    def __init__(self, alpha, beta, batch_size=1, *args, **kwargs):
        """
        Constructor.
        """
        if alpha.dim() == 1 and beta.dim() == 1:
            self.alpha = alpha.expand(batch_size, 0)
            self.beta = beta.expand(batch_size, 0)
        else:
            self.alpha = alpha
            self.beta = beta
        self.k = alpha
        self.theta = torch.pow(beta, -1.0)
        self.reparametrized = False
        super(Gamma, self).__init__(*args, **kwargs)

    def sample(self):
        """
        un-reparameterized sampler.
        """
        x = pyro.device(Variable(torch.Tensor([spr.gamma.rvs(
            self.alpha.data.cpu().numpy(), scale=self.theta.data.cpu().numpy())])))
        return x

    def log_pdf(self, x):
        """
        gamma log-likelihood
        """
        ll_1 = -self.beta * x
        ll_2 = (self.alpha - pyro.ones(self.alpha.size())) * torch.log(x)
        ll_3 = self.alpha * torch.log(self.beta)
        ll_4 = - log_gamma(self.alpha)
        return ll_1 + ll_2 + ll_3 + ll_4

    def batch_log_pdf(self, x, batch_size=1):
        """
        gamma log-likelihood
        """
        if x.dim() == 1 and self.beta.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = -self.beta * x
        ll_2 = (self.alpha - pyro.ones(x.size())) * torch.log(x)
        ll_3 = self.alpha * torch.log(self.beta)
        ll_4 = - log_gamma(self.alpha)
        return ll_1 + ll_2 + ll_3 + ll_4
