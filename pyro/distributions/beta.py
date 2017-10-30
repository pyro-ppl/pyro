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

    def __init__(self, alpha, beta, batch_size=None, *args, **kwargs):
        """
        Params:
          `alpha` - alpha
          `beta` - beta
        """
        self.alpha = alpha
        self.beta = beta
        if alpha.size() != beta.size():
            raise ValueError("Expected alpha.size() == beta.size(), but got {} vs {}"
                             .format(alpha.size(), beta.size()))
        if alpha.dim() == 1 and beta.dim() == 1 and batch_size is not None:
            self.alpha = alpha.expand(batch_size, alpha.size(0))
            self.beta = beta.expand(batch_size, beta.size(0))
        super(Beta, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        event_dim = 1
        alpha = self.alpha
        if x is not None and x.size() != alpha.size():
            alpha = self.alpha.expand_as(x)
        return alpha.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.alpha.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        """
        Un-reparameterizeable sampler.
        """
        np_sample = spr.beta.rvs(self.alpha.data.cpu().numpy(),
                                 self.beta.data.cpu().numpy())
        if isinstance(np_sample, numbers.Number):
            np_sample = [np_sample]
        x = Variable(torch.Tensor(np_sample).type_as(self.alpha.data))
        x = x.expand(self.shape())
        return x

    def batch_log_pdf(self, x):
        alpha = self.alpha.expand(self.shape(x))
        beta = self.beta.expand(self.shape(x))
        one = Variable(torch.ones(x.size()).type_as(alpha.data))
        ll_1 = (alpha - one) * torch.log(x)
        ll_2 = (beta - one) * torch.log(one - x)
        ll_3 = log_gamma(alpha + beta)
        ll_4 = -log_gamma(alpha)
        ll_5 = -log_gamma(beta)
        batch_log_pdf = torch.sum(ll_1 + ll_2 + ll_3 + ll_4 + ll_5, -1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        return self.alpha / (self.alpha + self.beta)

    def analytic_var(self):
        return torch.pow(self.analytic_mean(), 2.0) * self.beta / \
            (self.alpha * (self.alpha + self.beta + Variable(torch.ones([1]))))
