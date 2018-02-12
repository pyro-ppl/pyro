from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class TransformedDistribution(TorchDistribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, [f])
        log p(Y) = log p(X) + log det (dX/dY)
    """
    stateful = True  # often true because transforms may cache intermediate results

    def __init__(self, base_dist, transforms, *args, **kwargs):
        if not isinstance(base_dist, TorchDistribution):
            raise TypeError('Only TorchDistributions can be transformed by TransformedDistribution')
        self.base_dist = base_dist
        torch_dist = torch.distributions.TransformedDistribution(base_dist.torch_dist, transforms)
        super(TransformedDistribution, self).__init__(torch_dist, *args, **kwargs)

    @property
    def transforms(self):
        return self.torch_dist.transforms
