import scipy.stats as spr
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma


class Gamma(Distribution):
    """
    :param shape:  *(real (0, Infinity))*
    :param scale:  *(real (0, Infinity))*

    Gamma distribution parameterized by alpha and beta
    """

    def _sanitize_input(self, alpha, beta):
        if alpha is not None:
            # stateless distribution
            return alpha, beta
        elif self.alpha is not None:
            # stateful distribution
            return self.alpha, self.beta
        else:
            raise ValueError("Parameter(s) were None")

    def __init__(self, alpha=None, beta=None, batch_size=1, *args, **kwargs):
        """
        Params:
          `alpha` - alpha
          `beta` - beta
        """
        self.alpha = alpha
        self.beta = beta
        if alpha is not None:
            if alpha.dim() == 1 and beta.dim() == 1:
                self.alpha = alpha.expand(batch_size, alpha.size(0))
                self.beta = beta.expand(batch_size, beta.size(0))
        super(Gamma, self).__init__(*args, **kwargs)

    def sample(self, alpha=None, beta=None, *args, **kwargs):
        """
        un-reparameterized sampler.
        """

        alpha, beta = self._sanitize_input(alpha, beta)
        theta = torch.pow(beta, -1.0)
        x = Variable(torch.Tensor([spr.gamma.rvs(
            alpha.data.cpu().numpy(), scale=theta.data.cpu().numpy())])
            .type_as(alpha.data))
        return x

    def batch_log_pdf(self, x, alpha=None, beta=None, batch_size=1, *args, **kwargs):
        alpha, beta = self._sanitize_input(alpha, beta)
        assert alpha.dim() == beta.dim()
        if x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        if alpha.size() != x.size():
            alpha = alpha.expand_as(x)
            beta = beta.expand_as(x)
        ll_1 = - beta * x
        ll_2 = (alpha - pyro.ones(x.size())) * torch.log(x)
        ll_3 = alpha * torch.log(beta)
        ll_4 = - log_gamma(alpha)
        return ll_1 + ll_2 + ll_3 + ll_4

    def analytic_mean(self, alpha=None, beta=None):
        alpha, beta = self._sanitize_input(alpha, beta)
        return alpha / beta

    def analytic_var(self, alpha=None, beta=None):
        alpha, beta = self._sanitize_input(alpha, beta)
        return alpha / torch.pow(beta, 2.0)
