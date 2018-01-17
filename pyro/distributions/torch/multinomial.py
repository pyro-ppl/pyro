from __future__ import absolute_import, division, print_function

import numbers

import torch
from torch.autograd import Variable

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Multinomial(TorchDistribution):
    """
    Multinomial distribution.

    Distribution over counts for `n` independent `Categorical(ps)` trials.

    This is often used in conjunction with `torch.nn.Softmax` to ensure
    probabilites `ps` are normalized.

    :param torch.autograd.Variable ps: Probabilities (real). Should be positive
        and should normalized over the rightmost axis.
    :param int n: Number of trials. Should be positive.
    """
    enumerable = True

    def __init__(self, ps, n, *args, **kwargs):
        if isinstance(n, Variable):
            n = n.data
        if not isinstance(n, numbers.Number):
            if n.max() != n.min():
                raise NotImplementedError('inhomogeneous n is not supported')
            n = n.view(-1)[0]
        n = int(n)
        torch_dist = torch.distributions.Multinomial(n, probs=ps)
        x_shape = ps.shape
        event_dim = 1
        super(Multinomial, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

    def batch_log_pdf(self, x):
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_pxs = self.torch_dist.log_prob(x)
        batch_log_pdf = log_pxs.view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf
