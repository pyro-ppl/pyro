import scipy.stats as spr
import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution
from pyro.util import log_gamma


class Beta(Distribution):
    """
    :param a: shape *(real (0, Infinity))*
    :param b: shape *(real (0, Infinity))*

    Univariate beta distribution parameterized by alpha and beta
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
            if alpha.dim() != beta.dim():
                raise ValueError("Alpha and beta need to have the same dimensions.")
            if alpha.dim() == 1 and beta.dim() == 1:
                self.alpha = alpha.expand(batch_size, alpha.size(0))
                self.beta = beta.expand(batch_size, beta.size(0))
            else:
                self.alpha = alpha
                self.beta = beta
        self.reparameterized = False
        super(Beta, self).__init__(*args, **kwargs)

    def sample(self, alpha=None, beta=None, *args, **kwargs):
        """
        Un-reparameterizeable sampler.
        """
        _alpha, _beta = self._sanitize_input(alpha, beta)
        x = Variable(torch.Tensor(
            [spr.beta.rvs(_alpha.data.cpu().numpy(), _beta.data.cpu().numpy())])
            .type_as(_alpha.data))
        return x

    def log_pdf(self, x, alpha=None, beta=None, *args, **kwargs):
        """
        Beta log-likelihood
        """
        _alpha, _beta = self._sanitize_input(alpha, beta)
        one = Variable(torch.ones(_alpha.size()).type_as(_alpha.data))
        ll_1 = (_alpha - one) * torch.log(x)
        ll_2 = (_beta - one) * torch.log(one - x)
        ll_3 = log_gamma(_alpha + _beta)
        ll_4 = -log_gamma(_alpha)
        ll_5 = -log_gamma(_beta)
        return ll_1 + ll_2 + ll_3 + ll_4 + ll_5

    def batch_log_pdf(self, x, alpha=None, beta=None, batch_size=1, *args, **kwargs):
        _alpha, _beta = self._sanitize_input(alpha, beta)
        if x.dim() == 1 and _beta.dim() == 1 and batch_size == 1:
            return self.log_pdf(x. _alpha, _beta)
        elif x.dim() == 1:
            x = x.expand(batch_size, x.size(0))
        one = Variable(torch.ones(x.size()).type_as(_alpha.data))
        ll_1 = (_alpha - one) * torch.log(x)
        ll_2 = (_beta - one) * torch.log(one - x)
        ll_3 = log_gamma(_alpha + _beta)
        ll_4 = -log_gamma(_alpha)
        ll_5 = -log_gamma(_beta)
        return ll_1 + ll_2 + ll_3 + ll_4 + ll_5
