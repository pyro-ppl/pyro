from __future__ import absolute_import, division, print_function

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

    def __init__(self, alpha, batch_size=None, *args, **kwargs):
        """
        :param alpha: A vector of concentration parameters.
        :type alpha: None or a torch.autograd.Variable of a torch.Tensor of dimension 1 or 2.
        :param int batch_size: DEPRECATED.
        """
        self.alpha = alpha
        if alpha.dim() not in (1, 2):
            raise ValueError("Parameter alpha must be either 1 or 2 dimensional.")
        if alpha.dim() == 1 and batch_size is not None:
            self.alpha = alpha.expand(batch_size, alpha.size(0))
        super(Dirichlet, self).__init__(*args, **kwargs)

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
        return self.alpha.size()[-1:]

    def sample(self):
        """
        Draws either a single sample (if alpha.dim() == 1), or one sample per param (if alpha.dim() == 2).

        (Un-reparameterized).

        :param torch.autograd.Variable alpha:
        """
        alpha_np = self.alpha.data.cpu().numpy()
        if self.alpha.dim() == 1:
            x_np = spr.dirichlet.rvs(alpha_np)[0]
        else:
            x_np = np.empty_like(alpha_np)
            for i in range(alpha_np.shape[0]):
                x_np[i, :] = spr.dirichlet.rvs(alpha_np[i, :])[0]
        x = Variable(type(self.alpha.data)(x_np))
        return x

    def batch_log_pdf(self, x):
        """
        Evaluates log probability density over one or a batch of samples.

        Each of alpha and x can be either a single value or a batch of values batched along dimension 0.
        If they are both batches, their batch sizes must agree.
        In any case, the rightmost size must agree.

        :param torch.autograd.Variable x: A value (if x.dim() == 1) or or batch of values (if x.dim() == 2).
        :param alpha: A vector of concentration parameters.
        :type alpha: torch.autograd.Variable or None.
        :return: log probability densities of each element in the batch.
        :rtype: torch.autograd.Variable of torch.Tensor of dimension 1.
        """
        alpha = self.alpha.expand(self.shape(x))
        x_sum = torch.sum(torch.mul(alpha - 1, torch.log(x)), -1)
        beta = log_beta(alpha)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return (x_sum - beta).contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        sum_alpha = torch.sum(self.alpha)
        return self.alpha / sum_alpha

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        sum_alpha = torch.sum(self.alpha)
        return self.alpha * (sum_alpha - self.alpha) / (torch.pow(sum_alpha, 2) * (1 + sum_alpha))
