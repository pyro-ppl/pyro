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
        torch_dist = torch.distributions.TransformedDistribution(base_dist, transforms)
        super(TransformedDistribution, self).__init__(torch_dist, *args, **kwargs)

    @property
    def base_dist(self):
        return self.torch_dist.base_dist

    @property
    def transforms(self):
        return self.torch_dist.transforms

    @property
    def reparameterized(self):
        return self.base_dist.reparameterized

    # We need to override .sample() because PyTorch .sample() is detached.
    def sample(self, sample_shape=torch.Size()):
        x = self.base_dist.sample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x
