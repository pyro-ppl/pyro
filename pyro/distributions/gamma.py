import scipy.stats as spr
import torch
from torch.autograd import Variable
import pyro
from pyro.distributions.distribution import Distribution
from pyro.util import log_gamma


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
        if x.dim() == 1 and self.beta.dim() == 1 and batch_size == 1:
            return self.log_pdf(x)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = -self.beta * x
        ll_2 = (self.alpha - pyro.ones(x.size())) * torch.log(x)
        ll_3 = self.alpha * torch.log(self.beta)
        ll_4 = - log_gamma(self.alpha)
        return ll_1 + ll_2 + ll_3 + ll_4
