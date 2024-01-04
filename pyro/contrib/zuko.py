# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
This file contains helpers to use `Zuko <https://zuko.readthedocs.io/>`_-based
normalizing flows within Pyro piplines.

Accompanying tutorials can be found at `tutorial/svi_flow_guide.ipynb` and
`tutorial/vae_flow_prior.ipynb`.
"""

import torch
from torch import Size, Tensor

import pyro


class Zuko2Pyro(pyro.distributions.TorchDistribution):
    r"""Wraps a Zuko distribution as a Pyro distribution.

    :param dist: A distribution instance.
    :type dist: torch.distributions.Distribution

    Example:
        >>> flow = zuko.flows.MAF(features=5)
        >>> dist = Zuko2Pyro(flow())
        >>> dist((2, 3)).shape
        torch.Size([2, 3, 5])
        >>> x = pyro.sample("x", dist)
    """

    def __init__(self, dist: torch.distributions.Distribution):
        self.dist = dist
        self.cache = {}

    @property
    def has_rsample(self) -> bool:
        return self.dist.has_rsample

    @property
    def event_shape(self) -> Size:
        return self.dist.event_shape

    @property
    def batch_shape(self) -> Size:
        return self.dist.batch_shape

    def __call__(self, shape: Size = ()) -> Tensor:
        if hasattr(self.dist, "rsample_and_log_prob"):  # fast sampling + scoring
            x, self.cache[x] = self.dist.rsample_and_log_prob(shape)
        elif self.has_rsample:
            x = self.dist.rsample(shape)
        else:
            x = self.dist.sample(shape)

        return x

    def log_prob(self, x: Tensor) -> Tensor:
        if x in self.cache:
            return self.cache[x]
        else:
            return self.dist.log_prob(x)

    def expand(self, *args, **kwargs):
        return Zuko2Pyro(self.dist.expand(*args, **kwargs))
