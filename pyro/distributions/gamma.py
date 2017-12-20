from __future__ import absolute_import, division, print_function

import numbers

import scipy.stats as spr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.torch_wrapper import TorchDistribution, torch_wrapper
from pyro.distributions.util import broadcast_shape, log_gamma


class Gamma(Distribution):
    """
    Gamma distribution parameterized by `alpha` and `beta`.

    This is often used in conjunction with `torch.nn.Softplus` to ensure
    `alpha` and `beta` parameters are positive.

    :param torch.autograd.Variable alpha: Shape parameter. Should be positive.
    :param torch.autograd.Variable beta: Shape parameter. Should be positive.
        Shouldb be the same shape as `alpha`.
    """

    def __init__(self, alpha, beta, batch_size=None, *args, **kwargs):
        if alpha.size() != beta.size():
            raise ValueError("Expected alpha.size() == beta.size(), but got {} vs {}".format(alpha.size(), beta.size()))
        self.alpha = alpha
        self.beta = beta
        if alpha.dim() == 1 and beta.dim() == 1 and batch_size is not None:
            self.alpha = alpha.expand(batch_size, alpha.size(0))
            self.beta = beta.expand(batch_size, beta.size(0))
        super(Gamma, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        alpha = self.alpha
        if x is not None:
            if x.size()[-event_dim] != alpha.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.alpha.size()[-1], but got {} vs {}".format(
                                     x.size(-1), alpha.size(-1)))
            try:
                alpha = self.alpha.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `alpha` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(alpha.size(), x.size(), str(e)))
        return alpha.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.alpha.size()[-event_dim:]

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        theta = torch.pow(self.beta, -1.0)
        np_sample = spr.gamma.rvs(self.alpha.data.cpu().numpy(), scale=theta.data.cpu().numpy())
        if isinstance(np_sample, numbers.Number):
            np_sample = [np_sample]
        x = Variable(torch.Tensor(np_sample).type_as(self.alpha.data))
        x = x.expand(self.shape())
        return x

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        alpha = self.alpha.expand(self.shape(x))
        beta = self.beta.expand(self.shape(x))
        ll_1 = -beta * x
        ll_2 = (alpha - 1.0) * torch.log(x)
        ll_3 = alpha * torch.log(beta)
        ll_4 = -log_gamma(alpha)
        log_pdf = torch.sum(ll_1 + ll_2 + ll_3 + ll_4, -1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.alpha / self.beta

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return self.alpha / torch.pow(self.beta, 2.0)


class TorchGamma(TorchDistribution):
    """
    Compatibility wrapper around
    `torch.distributions.Gamma <http://pytorch.org/docs/master/_modules/torch/distributions.html#Gamma>`_
    """
    reparameterized = True

    def __init__(self, alpha, beta, log_pdf_mask=None, *args, **kwargs):
        torch_dist = torch.distributions.Gamma(alpha, beta)
        super(TorchGamma, self).__init__(torch_dist, log_pdf_mask, *args, **kwargs)
        self._param_shape = broadcast_shape(alpha.size(), beta.size(), strict=True)

    def batch_shape(self, x=None):
        x_shape = [] if x is None else x.size()
        shape = torch.Size(broadcast_shape(x_shape, self._param_shape, strict=True))
        return shape[:-1]

    def event_shape(self):
        return self._param_shape[-1:]


@torch_wrapper(Gamma)
def WrapGamma(alpha, beta, batch_size=None, log_pdf_mask=None, *args, **kwargs):
    if not hasattr(torch.distributions, 'Gamma'):
        raise NotImplementedError('Missing class torch.distribution.Gamma')
    elif batch_size is not None:
        raise NotImplementedError('Unsupported args')
    return TorchGamma(alpha, beta, log_pdf_mask=log_pdf_mask, *args, **kwargs)
