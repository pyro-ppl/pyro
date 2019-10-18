import torch
import torch.nn as nn
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.transforms.utils import clamp_preserve_gradients

import numpy as np


@copy_docs_from(TransformModule)
class GlowFlow(TransformModule):
    """
    An implementation of Glow: Generative Flow with Invertible 1×1 Convolutions, using Eq (9) from Kingma Et Al., 2018.
    Together with `TransformedDistribution` and `AffineCoupling` this provides a way to sample images from a latent distirbution (à la VAE).
    Example usage:
    >>> import torch
    >>> c = 3 # assume 3 x 32 x 32 RGB inputs
    >>> base_dist = dist.Normal(torch.zeros(c*32*32), torch.ones(c*32*32))
    >>> glow = GlowFlow(torch.nn.Conv2d(c,c,1))
    >>> pyro.module("my_glow", glow)  # doctest: +SKIP
    >>> glow_dist = dist.TransformedDistribution(base_dist, [ReshapeTransform([32,32,c]),glow])
    >>> glow_dist.sample(torch.Size([1])).shape  # doctest: +SKIP
        torch.Size([1, 3, 32, 32])

    :param cnn: a convolutional neural network with 1x1 kernel size
    :type cnn: nn.Module
    :param initialized: if False, use Kingma's orthogonal initialization
    :type initialized: bool
    References:
    1. Glow: Generative Flow with Invertible 1×1 Convolutions [arXiv:1807.03039]
    Diederik P. Kingma, Prafulla Dhariwal
    2. Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 0
    autoregressive = False

    def __init__(self, cnn, initialized=True):
        super(GlowFlow, self).__init__(cache_size=1)
        self.cnn = cnn
        self._cached_log_det = None
        self.initialized = initialized

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor
        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        h,w,c = x.shape[1:]
        if not self.initialized:
            w_init = torch.FloatTensor(np.linalg.qr(np.random.randn(*[c,c]))[0]) # c x c
            self.cnn.weight.data[:,:,0,0] = w_init
            self.initialized = True

        log_det = h * w * torch.log(torch.abs(torch.det(self.cnn.weight)))
        self._cached_log_det = log_det

        y = self.cnn(x)
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """
        h,w,c = y.shape[1:]
        if not self.initialized:
            w_init = torch.FloatTensor(np.linalg.qr(np.random.randn(*[c,c]))[0]) # c x c
            self.cnn.weight.data[:,:,0,0] = w_init
            self.initialized = True

        log_det = - (h * w * torch.log(torch.abs(torch.det(self.cnn.weight))) )
        self._cached_log_det = log_det

        self.cnn.weight = torch.inverse(self.cnn.weight)

        x = self.cnn(y)
        return x

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        if self._cached_log_det is not None:
            log_det = self._cached_log_det
        else:
            log_det = h * w * torch.log(torch.abs(torch.det(self.cnn.weight)))
        return log_det.sum(-1)