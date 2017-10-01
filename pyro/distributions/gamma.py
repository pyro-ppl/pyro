import scipy.stats as spr
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions.distribution import Distribution
from pyro.util import log_gamma


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
        self.reparameterized = False
        super(Gamma, self).__init__(*args, **kwargs)

    def sample(self, alpha=None, beta=None, *args, **kwargs):
        """
        un-reparameterized sampler.
        """

        _alpha, _beta = self._sanitize_input(alpha, beta)
        _theta = torch.pow(_beta, -1.0)
        x = Variable(torch.Tensor([spr.gamma.rvs(
            _alpha.data.numpy(), scale=_theta.data.numpy())])
            .type_as(_alpha.data))
        return x

    def log_pdf(self, x, alpha=None, beta=None, *args, **kwargs):
        """
        gamma log-likelihood
        """
        _alpha, _beta = self._sanitize_input(alpha, beta)
        ll_1 = -_beta * x
        ll_2 = (_alpha - pyro.ones(_alpha.size())) * torch.log(x)
        ll_3 = _alpha * torch.log(_beta)
        ll_4 = - log_gamma(_alpha)
        return ll_1 + ll_2 + ll_3 + ll_4

    def batch_log_pdf(self, x, alpha=None, beta=None, batch_size=1, *args, **kwargs):
        _alpha, _beta = self._sanitize_input(alpha, beta)
        if x.dim() == 1 and _beta.dim() == 1 and batch_size == 1:
            return self.log_pdf(x, _alpha, _beta)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        ll_1 = -_beta * x
        ll_2 = (_alpha - pyro.ones(x.size())) * torch.log(x)
        ll_3 = _alpha * torch.log(_beta)
        ll_4 = - log_gamma(_alpha)
        return ll_1 + ll_2 + ll_3 + ll_4

    def analytic_mean(self, alpha=None, beta=None):
        _alpha, _beta = self._sanitize_input(alpha, beta)
        return _alpha / _beta

    def analytic_var(self, alpha=None, beta=None):
        _alpha, _beta = self._sanitize_input(alpha, beta)
        return _alpha / torch.pow(_beta, 2.0)
