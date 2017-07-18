import scipy.stats as spr
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution
from pyro.util import log_gamma


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
