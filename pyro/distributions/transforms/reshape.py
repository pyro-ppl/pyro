import torch
from torch.distributions.utils import lazy_property
from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.distributions.util import copy_docs_from


@copy_docs_from(Transform)
class ReshapeTransform(Transform):
    """
    A bijection that reshapes the inputs into an arbitrary-size Tensor. In conjunction with `GlowFlow`, this can be used
    to apply `nn.Conv2d` to any supported distribution.

    Example usage:
    >>> import torch
    >>> c = 3 # assume 3 x 32 x 32 RGB inputs
    >>> base_dist = dist.Normal(torch.zeros(c*32*32), torch.ones(c*32*32))
    >>> glow = GlowFlow(torch.nn.Conv2d(c,c,1))
    >>> pyro.module("my_glow", glow)  # doctest: +SKIP
    >>> glow_dist = dist.TransformedDistribution(base_dist, [ReshapeTransform([32,32,c]),glow])
    >>> glow_dist.sample(torch.Size([1])).shape  # doctest: +SKIP
        torch.Size([1, 3, 32, 32])

    :param shape: the desired shape of y, for the bijection x=>y.
    :type shape: list

    """

    codomain = constraints.real
    bijective = True
    event_dim = 1
    volume_preserving = True

    def __init__(self, shape):
        super(ReshapeTransform, self).__init__(cache_size=1)

        self._cached_forward_shape = shape

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous transform)
        """
        self._cached_inverse_shape = x.shape[1:]
        return x.view([x.size(0)]+self._cached_forward_shape)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """
        self._cached_forward_shape = y.shape[1:]
        return y.view([y.size(0)]+self._cached_inverse_shape)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e. log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])).
        Note that this type of transform is not autoregressive, so the log Jacobian is not the sum of the previous
        expression. However, it turns out it's always 0 (since the determinant is -1 or +1), and so returning a
        vector of zeros works.
        """

        return torch.zeros(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device)
