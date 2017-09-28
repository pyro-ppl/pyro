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

        _alpha = self._sanitize_input(alpha)
        # _alpha = Variable(torch.Tensor([[1,2],[3,4]]))
        x = Variable(torch.Tensor(spr.dirichlet.rvs(
                     _alpha.data.numpy()))
                     .type_as(_alpha.data)).squeeze(0)
        return x

    def log_pdf(self, x, alpha=None, *args, **kwargs):
        _alpha = self._sanitize_input(alpha)
        x_sum = torch.sum(torch.mul(_alpha - 1, torch.log(x)))
        beta = log_beta(_alpha)
        return x_sum - beta

    def batch_log_pdf(self, x, alpha=None, batch_size=1, *args, **kwargs):
        _alpha = self._sanitize_input(alpha)
        if x.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, _alpha)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        x_sum = torch.sum(torch.mul(_alpha - 1, torch.log(x)), 1)
        beta = log_beta(_alpha)
        return x_sum - beta

    def analytic_mean(self, alpha):
        _alpha = self._sanitize_input(alpha)
        _sum_alpha = torch.sum(_alpha)
        return _alpha / _sum_alpha

    def analytic_var(self, alpha):
        _alpha = self._sanitize_input(alpha)
        _sum_alpha = torch.sum(_alpha)
        return _alpha * (_sum_alpha - _alpha) / (torch.pow(_sum_alpha, 2) * (1 + _sum_alpha))
