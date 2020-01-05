import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform

from pyro.distributions.util import copy_docs_from


@copy_docs_from(Transform)
class ReshapeEvent(Transform):
    """
    A bijection that reshapes the inputs into an arbitrary-size Tensor. The primary intended use of this transform
    is to allow interoperatability between transforms on 1D vector random variables and 2D image random variables.

    Note that this transform may convert batch dimensions into event dimensions, depending on the input/output shapes.
    The event dimension of the output of this transformation will be the length of the to_event_shape argument.

    Example usage:
    >>> import torch
    >>> c = 3 # assume 3 x 32 x 32 RGB inputs
    >>> base_dist = dist.Normal(torch.zeros(c*32*32), torch.ones(c*32*32))
    >>> glow = GlowFlow(torch.nn.Conv2d(c,c,1))
    >>> pyro.module("my_glow", glow)  # doctest: +SKIP
    >>> glow_dist = dist.TransformedDistribution(base_dist, [ReshapeTransform([32,32,c]),glow])
    >>> glow_dist.sample(torch.Size([1])).shape  # doctest: +SKIP
        torch.Size([1, 3, 32, 32])

    :param from_event_shape: the shape of the rightmost dimensions of x to reshape, for the bijection x=>y.
    :type shape: list
    :param to_event_shape: the desired shape of the rightmost dimensions of y.
    :type shape: list

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    volume_preserving = True

    def __init__(self, from_event_shape, to_event_shape):
        super(ReshapeEvent, self).__init__(cache_size=1)

        self.from_event_shape = torch.Size(from_event_shape)
        self.to_event_shape = torch.Size(to_event_shape)
        self.from_event_dim = len(from_event_shape)
        self.event_dim = len(to_event_shape)

        # Check that dimensions haven't been introduced or removed
        if torch.tensor(self.from_event_shape).sum() != torch.tensor(self.to_event_shape).sum():
            raise ValueError(
                "There is a mismatch between the input shape {} and the output shape {}!".format(
                    from_event_shape, to_event_shape))

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous transform)
        """

        shape = x.shape[:-self.from_event_dim] + self.to_event_shape
        return x.view(shape)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """

        shape = y.shape[:-self.event_dim] + self.from_event_shape
        return y.view(shape)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e. log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])).
        Note that this type of transform is not autoregressive, so the log Jacobian is not the sum of the previous
        expression. However, it turns out it's always 0 (since the determinant is -1 or +1), and so returning a
        vector of zeros works.
        """

        return torch.zeros(x.size()[:-self.from_event_dim], dtype=x.dtype, layout=x.layout, device=x.device)
