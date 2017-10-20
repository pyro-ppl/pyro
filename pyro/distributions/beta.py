import numbers

import scipy.stats as spr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_gamma


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

    def __init__(self, alpha=None, beta=None, batch_size=None, *args, **kwargs):
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
            if alpha.dim() == 1 and beta.dim() == 1 and batch_size is not None:
                self.alpha = alpha.expand(batch_size, alpha.size(0))
                self.beta = beta.expand(batch_size, beta.size(0))
        super(Beta, self).__init__(*args, **kwargs)

    def batch_shape(self, alpha=None, beta=None, *args, **kwargs):
        alpha, beta = self._sanitize_input(alpha, beta)
        event_dim = 1
        return alpha.size()[:-event_dim]

    def event_shape(self, alpha=None, beta=None, *args, **kwargs):
        alpha, beta = self._sanitize_input(alpha, beta)
        event_dim = 1
        return alpha.size()[-event_dim:]

    def sample(self, alpha=None, beta=None, *args, **kwargs):
        """
        Un-reparameterizeable sampler.
        """
        alpha, beta = self._sanitize_input(alpha, beta)
        np_sample = spr.beta.rvs(alpha.data.cpu().numpy(), beta.data.cpu().numpy())
        if isinstance(np_sample, numbers.Number):
            np_sample = [np_sample]
        x = Variable(torch.Tensor(np_sample).type_as(alpha.data))
        x = x.expand(self.shape(alpha, beta))
        return x

    def batch_log_pdf(self, x, alpha=None, beta=None, *args, **kwargs):
        alpha, beta = self._sanitize_input(alpha, beta)
        assert alpha.dim() == beta.dim()
        if alpha.size() != x.size():
            alpha = alpha.expand_as(x)
            beta = beta.expand_as(x)
        one = Variable(torch.ones(x.size()).type_as(alpha.data))
        ll_1 = (alpha - one) * torch.log(x)
        ll_2 = (beta - one) * torch.log(one - x)
        ll_3 = log_gamma(alpha + beta)
        ll_4 = -log_gamma(alpha)
        ll_5 = -log_gamma(beta)
        log_pdf = torch.sum(ll_1 + ll_2 + ll_3 + ll_4 + ll_5, -1)
        batch_log_pdf_shape = self.batch_shape(alpha, beta) + (1,)
        return log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self, alpha=None, beta=None):
        alpha, beta = self._sanitize_input(alpha, beta)
        return alpha / (alpha + beta)

    def analytic_var(self, alpha=None, beta=None):
        alpha, beta = self._sanitize_input(alpha, beta)
        return torch.pow(self.analytic_mean(alpha, beta), 2.0) * beta / \
            (alpha * (alpha + beta + Variable(torch.ones([1]))))
