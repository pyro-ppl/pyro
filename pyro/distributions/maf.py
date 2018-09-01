from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform
from torch.distributions import constraints

from pyro.distributions.util import copy_docs_from


def clamp_preserve_gradients(x, min, max):
    return x + (x.clamp(min, max) - x).detach()


@copy_docs_from(Transform)
class MaskedAutoregressiveFlow(Transform):
    """
    An implementation of a Masked Autoregressive Flow used to create rich distributions with tractable
    log-density calculation.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> maf = MaskedAutoregressiveFlow(AutoRegressiveNN(10, [40]))
    >>> maf_module = pyro.module("my_maf", maf.module)
    >>> maf_dist = dist.TransformedDistribution(base_dist, [maf])
    >>> maf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    IAF and MAF can be seen as the inverse of each other. I.e. MAF simple swaps the forward and inverse
    operations of IAF. This means that scoring an arbitrary sample is fast and sampling scales as O(D),
    where D is the input dimension, and so should be avoided for large dimensional uses.

    :param autoregressive_nn: an autoregressive neural network whose forward call returns a real-valued
        mean and logit-scale as a tuple
    :type autoregressive_nn: nn.Module
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_max_clip: float

    References:

    1. Masked Autoregressive Flow for Density Estimation  arXiv:1705.07057v4 [stat.ML]
    George Papamakarios, Theo Pavlakou, Iain Murray

    """

    codomain = constraints.real

    def __init__(self, autoregressive_nn, log_scale_min_clip=-5., log_scale_max_clip=3.):
        super(MaskedAutoregressiveFlow, self).__init__()
        self.module = nn.Module()
        self.module.arn = autoregressive_nn
        self._intermediates_cache = {}
        self.add_inverse_to_cache = True
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    @property
    def arn(self):
        """
        :rtype: pyro.nn.AutoRegressiveNN

        Return the AutoRegressiveNN associated with the InverseAutoregressiveFlow
        """
        return self.module.arn

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        y_size = x.size()[:-1]
        perm = self.module.arn.permutation
        input_dim = x.size(-1)
        y = [torch.zeros(y_size, device=x.device)] * input_dim

        # NOTE: Sampling is an expensive operation that scales in the dimension of the input
        for idx in perm:
            mean, log_scale = self.module.arn(torch.stack(y, dim=-1))
            inverse_scale = torch.exp(-clamp_preserve_gradients(
                log_scale[..., idx], min=self.log_scale_min_clip, max=self.log_scale_max_clip))
            mean = mean[..., idx]
            y[idx] = (x[..., idx] - mean) * inverse_scale

        y = torch.stack(y, dim=-1)
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """
        mean, log_scale = self.module.arn(y)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        scale = torch.exp(log_scale)

        x = scale * y + mean
        self._add_intermediate_to_cache(log_scale, y, 'log_scale')
        return x

    def _add_intermediate_to_cache(self, intermediate, y, name):
        """
        Internal function used to cache intermediate results computed during the inverse call
        """
        # assert((y, name) not in self._intermediates_cache),\
        #    "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[(y, name)] = intermediate

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        if (y, 'log_scale') in self._intermediates_cache:
            log_scale = self._intermediates_cache.pop((y, 'log_scale'))
        else:
            _, log_scale = self.module.arn(x)
            log_scale = clamp_preserve_gradients(log_scale, min=self.log_scale_min_clip, max=self.log_scale_max_clip)
        return -log_scale
