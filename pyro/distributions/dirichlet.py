import numpy as np
import scipy.stats as spr
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import log_beta


class Dirichlet(Distribution):
    """
    Dirichlet distribution parameterized by a vector `alpha`.

    Dirichlet is a multivariate generalization of the Beta distribution.

    :param alpha:  *(real (0, Infinity))*
    """

    def _sanitize_input(self, alpha):
        if alpha is not None:
            # stateless distribution
            return alpha
        if self.alpha is not None:
            # stateful distribution
            return self.alpha
        raise ValueError("Parameter(s) were None")

    def _expand_dims(self, x, alpha):
        """
        Expand to 2-dimensional tensors of the same shape.
        """
        if not isinstance(x, (torch.Tensor, Variable)):
            raise TypeError('Expected x a Tensor or Variable, got a {}'.format(type(x)))
        if not isinstance(alpha, Variable):
            raise TypeError('Expected alpha a Variable, got a {}'.format(type(alpha)))
        if x.dim() not in (1, 2):
            raise ValueError('Expected x.dim() in (1,2), actual: {}'.format(x.dim()))
        if alpha.dim() not in (1, 2):
            raise ValueError('Expected alpha.dim() in (1,2), actual: {}'.format(alpha.dim()))
        if x.size() != alpha.size():
            alpha = alpha.expand_as(x)
        return x, alpha

    def __init__(self, alpha=None, batch_size=None, *args, **kwargs):
        """
        :param alpha: A vector of concentration parameters.
        :type alpha: None or a torch.autograd.Variable of a torch.Tensor of dimension 1 or 2.
        :param int batch_size: DEPRECATED.
        """
        if alpha is None:
            self.alpha = None
        else:
            assert alpha.dim() in (1, 2)
            self.alpha = alpha
        super(Dirichlet, self).__init__(*args, **kwargs)

    def batch_shape(self, alpha=None):
        alpha = self._sanitize_input(alpha)
        return alpha.size()[:-1]

    def event_shape(self, alpha=None):
        alpha = self._sanitize_input(alpha)
        return alpha.size()[-1:]

    def sample(self, alpha=None):
        """
        Draws either a single sample (if alpha.dim() == 1), or one sample per param (if alpha.dim() == 2).

        (Un-reparameterized).

        :param torch.autograd.Variable alpha:
        """
        alpha = self._sanitize_input(alpha)
        if alpha.dim() not in (1, 2):
            raise ValueError('Expected alpha.dim() in (1,2), actual: {}'.format(alpha.dim()))
        alpha_np = alpha.data.cpu().numpy()
        if alpha.dim() == 1:
            x_np = spr.dirichlet.rvs(alpha_np)[0]
        else:
            x_np = np.empty_like(alpha_np)
            for i in range(alpha_np.shape[0]):
                x_np[i, :] = spr.dirichlet.rvs(alpha_np[i, :])[0]
        x = Variable(type(alpha.data)(x_np))
        return x

    def batch_log_pdf(self, x, alpha=None):
        """
        Evaluates log probabity density over one or a batch of samples.

        Each of alpha and x can be either a single value or a batch of values batched along dimension 0.
        If they are both batches, their batch sizes must agree.
        In any case, the rightmost size must agree.

        :param torch.autograd.Variable x: A value (if x.dim() == 1) or or batch of values (if x.dim() == 2).
        :param alpha: A vector of concentration parameters.
        :type alpha: torch.autograd.Variable or None.
        :return: log probability densities of each element in the batch.
        :rtype: torch.autograd.Variable of torch.Tensor of dimension 1.
        """
        alpha = self._sanitize_input(alpha)
        x, alpha = self._expand_dims(x, alpha)
        assert x.size() == alpha.size()
        x_sum = torch.sum(torch.mul(alpha - 1, torch.log(x)), -1)
        beta = log_beta(alpha)
        batch_log_pdf_shape = self.batch_shape(alpha) + (1,)
        return (x_sum - beta).contiguous().view(batch_log_pdf_shape)

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
