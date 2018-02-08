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
    Note that `n` need not be specified if only :meth:`log_pdf` is used, e.g.
    if this distribution is used only in :meth:`pyro.observe` statements.

    This is often used in conjunction with `torch.nn.Softmax` to ensure
    probabilites `ps` are normalized.

    :param torch.autograd.Variable ps: Probabilities (real). Should be positive
        and should normalized over the rightmost axis.
    :param int n: Optional number of trials. Should be positive. Defaults to 1.
        Note that this is ignored by `.log_pdf()` which infers a `n` from each
        event.
    """
    enumerable = True

    def __init__(self, ps, n=1, *args, **kwargs):
        if isinstance(n, Variable):
            n = n.data
        if not isinstance(n, numbers.Number):
            if n.max() != n.min():
                raise NotImplementedError('inhomogeneous n is not supported')
            n = n.view(-1)[0]
        n = int(n)
        torch_dist = torch.distributions.Multinomial(n, probs=ps)
        super(Multinomial, self).__init__(torch_dist, *args, **kwargs)
