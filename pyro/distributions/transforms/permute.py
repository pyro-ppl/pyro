from __future__ import absolute_import, division, print_function

import torch
from torch.distributions.transforms import Transform
from torch.distributions.utils import lazy_property
from torch.distributions import constraints

from pyro.distributions.util import copy_docs_from


@copy_docs_from(Transform)
class PermuteTransform(Transform):
    """
    A bijection that reorders the input dimensions, that is, multiplies the input by a permutation matrix.
    This is useful in between :class:`~pyro.distributions.InverseAutoregressiveFlow` transforms to increase the
    flexibility of the resulting distribution and stabilize learning. Whilst not being an autoregressive transform,
    the log absolute determinate of the Jacobian is easily calculable as 0. Note that reordering the input dimension
    between two layers of :class:`~pyro.distributions.InverseAutoregressiveFlow` is not equivalent to reordering
    the dimension inside the MADE networks that those IAFs use; using a PermuteTransform results in a distribution
    with more flexibility.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> from pyro.distributions import InverseAutoregressiveFlow, PermuteTransform
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iaf1 = InverseAutoregressiveFlow(AutoRegressiveNN(10, [40]))
    >>> ff = PermuteTransform(torch.randperm(10, dtype=torch.long))
    >>> iaf2 = InverseAutoregressiveFlow(AutoRegressiveNN(10, [40]))
    >>> iaf_dist = dist.TransformedDistribution(base_dist, [iaf1, ff, iaf2])
    >>> iaf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    :param permutation: a permutation ordering that is applied to the inputs.
    :type permutation: torch.LongTensor

    """

    codomain = constraints.real
    bijective = True
    event_dim = 1
    volume_preserving = True

    def __init__(self, permutation):
        super(PermuteTransform, self).__init__(cache_size=1)

        self.permutation = permutation

    @lazy_property
    def inv_permutation(self):
        result = torch.empty_like(self.permutation, dtype=torch.long)
        result[self.permutation] = torch.arange(self.permutation.size(0),
                                                dtype=torch.long,
                                                device=self.permutation.device)
        return result

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous transform)
        """

        return x[..., self.permutation]

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """

        return y[..., self.inv_permutation]

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e. log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])).
        Note that this type of transform is not autoregressive, so the log Jacobian is not the sum of the previous
        expression. However, it turns out it's always 0 (since the determinant is -1 or +1), and so returning a
        vector of zeros works.
        """

        return torch.zeros(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device)
