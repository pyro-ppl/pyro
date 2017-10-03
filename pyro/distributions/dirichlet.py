import scipy.stats as spr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.util import log_beta


class Dirichlet(Distribution):
    """
    :param alpha:  *(real (0, Infinity))*

    Dirichlet distribution parameterized by alpha. Dirichlet
    is a multivariate generalization of the Beta distribution
    """

    def _sanitize_input(self, alpha):
        if alpha is not None:
            # stateless distribution
            return alpha
        elif self.alpha is not None:
            # stateful distribution
            return self.alpha
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, alpha=None, batch_size=1, *args, **kwargs):
        """
        Params:
          `alpha` - alpha
        """
        self.alpha = alpha
        if alpha is not None:
            if alpha.dim() == 1:
                self.alpha = alpha.expand(batch_size, alpha.size(0))
        self.reparameterized = False
        super(Dirichlet, self).__init__(*args, **kwargs)

    def sample(self, alpha=None, *args, **kwargs):
        """
        un-reparameterized sampler.
        """

        alpha = self._sanitize_input(alpha)
        # alpha = Variable(torch.Tensor([[1,2],[3,4]]))
        x = Variable(torch.Tensor(spr.dirichlet.rvs(
                     alpha.data.numpy()))
                     .type_as(alpha.data)).squeeze(0)
        return x

    def log_pdf(self, x, alpha=None, *args, **kwargs):
        alpha = self._sanitize_input(alpha)
        x_sum = torch.sum(torch.mul(alpha - 1, torch.log(x)))
        beta = log_beta(alpha)
        return x_sum - beta

    def batch_log_pdf(self, x, alpha=None, batch_size=1, *args, **kwargs):
        alpha = self._sanitize_input(alpha)
        if x.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, alpha)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        x_sum = torch.sum(torch.mul(alpha - 1, torch.log(x)), 1)
        beta = log_beta(alpha)
        return x_sum - beta

    def analytic_mean(self, alpha):
        alpha = self._sanitize_input(alpha)
        sum_alpha = torch.sum(alpha)
        return alpha / sum_alpha

    def analytic_var(self, alpha):
        """
        :return: Analytic variance of the dirichlet distribution, with parameter alpha.
        :rtype: torch.autograd.Variable (Vector of the same size as alpha).
        """
        alpha = self._sanitize_input(alpha)
        sum_alpha = torch.sum(alpha)
        return alpha * (sum_alpha - alpha) / (torch.pow(sum_alpha, 2) * (1 + sum_alpha))
