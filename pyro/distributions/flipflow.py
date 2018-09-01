from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.distributions.transforms import Transform
from torch.distributions import constraints

from pyro.distributions.util import copy_docs_from


@copy_docs_from(Transform)
class FlipFlow(Transform):
    """
    A normalizing flow that reorders the input dimensions, that is, multiplies the input by a permutation matrix.
    This is useful in between IAF transforms to increase the flexibility of the resulting distribution and
    stabilize learning. Whilst not being an autoregressive flow, the log absolute determinate of the Jacobian is
    easily calculable as 0. Note that reordering the input dimension between two layers of IAF is not equivalent 
    to reordering the dimension inside the MADE networks that those IAFs use; using a FlipFlow results in a
    distribution with more flexibility.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iaf = InverseAutoregressiveFlow(AutoRegressiveNN(10, [40]))
    >>> iaf_module = pyro.module("my_iaf", iaf.module)
    >>> iaf_dist = dist.TransformedDistribution(base_dist, [iaf])
    >>> iaf_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])
    
    :param permutation: a optional permutation ordering that is applied to the inputs.
    :type permutation: torch.LongTensor

    """

    codomain = constraints.real

    def __init__(self, permutation):
        super(FlipFlow, self).__init__()

        self.permutation = permutation

        # Calculate the inverse permutation order

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """

        return x[...,self.permutation]

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise performs the inversion afresh.
        """

        # To invert y, we simply apply the same permutation again
        return y[...,self.permutation]

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e. log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])).
        Note that this type of flow is not autoregressive, so the log Jacobian is not the sum of the previous
        expression. However, it turns out it's always 0 (since the determinant is -1 or +1), and so returning a 
        vector of zeros works.
        """

        return torch.zeros_like(x)
