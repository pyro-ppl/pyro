from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints

from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.distributions.transforms.utils import clamp_preserve_gradients


@copy_docs_from(TransformModule)
class AffineCoupling(TransformModule):
    """
    An implementation of the affine coupling layer of RealNVP (Dinh et al., 2017) that uses the transformation,

        :math:`\\mathbf{y}_{1:d} = \\mathbf{x}_{1:d}`
        :math:`\\mathbf{y}_{(d+1):D} = \\mu + \\sigma\\odot\\mathbf{x}_{(d+1):D}`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs, e.g. :math:`\\mathbf{x}_{1:d}
    represents the first :math:`d` elements of the inputs, and :math:`\\mu,\\sigma` are shift and translation
    parameters calculated as the output of a function inputting only :math:`\\mathbf{x}_{1:d}`.

    That is, the first :math:`d` components remain unchanged, and the subsequent :math:`D-d` are shifted and
    translated by a function of the previous components.

    Together with `TransformedDistribution` this provides a way to create richer variational approximations.

    Example usage:

    >>> from pyro.nn import DenseNN
    >>> input_dim = 10
    >>> split_dim = 6
    >>> base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
    >>> hypernet = DenseNN(split_dim, [10*input_dim], [input_dim-split_dim, input_dim-split_dim])
    >>> flow = AffineCoupling(split_dim, hypernet)
    >>> pyro.module("my_flow", flow)  # doctest: +SKIP
    >>> flow_dist = dist.TransformedDistribution(base_dist, [flow])
    >>> flow_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    The inverse of the Bijector is required when, e.g., scoring the log density of a sample with
    `TransformedDistribution`. This implementation caches the inverse of the Bijector when its forward
    operation is called, e.g., when sampling from `TransformedDistribution`. However, if the cached value
    isn't available, either because it was overwritten during sampling a new value or an arbitary value is
    being scored, it will calculate it manually.

    This is an operation that scales as O(1), i.e. constant in the input dimension. So in general, it is cheap
    to sample *and* score (an arbitrary value) from AffineCoupling.

    :param split_dim: Zero-indexed dimension :math:`d` upon which to perform input/output split for transformation.
    :type split_dim: int
    :param hypernet: an autoregressive neural network whose forward call returns a real-valued
        mean and logit-scale as a tuple. The input should have final dimension split_dim and the output final
        dimension input_dim-split_dim for each member of the tuple.
    :type hypernet: callable
    :param log_scale_min_clip: The minimum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_min_clip: float
    :param log_scale_max_clip: The maximum value for clipping the log(scale) from the autoregressive NN
    :type log_scale_max_clip: float

    References:

    Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. ICLR 2017.

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 1

    def __init__(self, split_dim, hypernet, log_scale_min_clip=-5., log_scale_max_clip=3.):
        super(AffineCoupling, self).__init__(cache_size=1)
        self.split_dim = split_dim
        self.hypernet = hypernet
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        mean, log_scale = self.hypernet(x1)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        y1 = x1
        y2 = torch.exp(log_scale) * x2 + mean
        return torch.cat([y1, y2], dim=-1)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """
        y1, y2 = y[..., :self.split_dim], y[..., self.split_dim:]
        x1 = y1
        mean, log_scale = self.arn(x1)
        log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        self._cached_log_scale = log_scale

        x2 = (y2 - mean) * torch.exp(-log_scale)
        return torch.cat([x1, x2], dim=-1)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if self._cached_log_scale is not None and x is x_old and y is y_old:
            log_scale = self._cached_log_scale
        else:
            x1 = x[..., :self.split_dim]
            _, log_scale = self.hypernet(x1)
            log_scale = clamp_preserve_gradients(log_scale, self.log_scale_min_clip, self.log_scale_max_clip)
        return log_scale.sum(-1)
