from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.transforms import Transform

from pyro.distributions.util import copy_docs_from
from pyro.nn import AutoRegressiveNN


@copy_docs_from(Transform)
class InverseAutoregressiveFlow(Transform):
    """
    An implementation of an Inverse Autoregressive Flow. Together with the `TransformedDistribution` this
    provides a way to create richer variational approximations.

    Example usage::

    >>> base_dist = Normal(...)
    >>> iaf = InverseAutoregressiveFlow(...)
    >>> pyro.module("my_iaf", iaf.module)
    >>> iaf_dist = TransformedDistribution(base_dist, [iaf])

    Note that this implementation is only meant to be used in settings where the inverse of the Bijector
    is never explicitly computed (rather the result is cached from the forward call). In the context of
    variational inference, this means that the InverseAutoregressiveFlow should only be used in the guide,
    i.e. in the variational distribution. In other contexts the inverse could in principle be computed but
    this would be a (potentially) costly computation that scales with the dimension of the input (and in
    any case support for this is not included in this implementation).

    :param input_dim: dimension of input
    :type input_dim: int
    :param hidden_dim: hidden dimension (number of hidden units)
    :type hidden_dim: int
    :param sigmoid_bias: bias on the hidden units fed into the sigmoid; default=`2.0`
    :type sigmoid_bias: float
    :param permutation: whether the order of the inputs should be permuted (by default the conditional
        dependence structure of the autoregression follows the sequential order)
    :type permutation: bool

    References:

    1. Improving Variational Inference with Inverse Autoregressive Flow [arXiv:1606.04934]
    Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling

    2. Variational Inference with Normalizing Flows [arXiv:1505.05770]
    Danilo Jimenez Rezende, Shakir Mohamed

    3. MADE: Masked Autoencoder for Distribution Estimation [arXiv:1502.03509]
    Mathieu Germain, Karol Gregor, Iain Murray, Hugo Larochelle
    """

    def __init__(self, input_dim, hidden_dim, sigmoid_bias=2.0, permutation=None):
        super(InverseAutoregressiveFlow, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.arn = AutoRegressiveNN(input_dim, hidden_dim, output_dim_multiplier=2, permutation=permutation)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_bias = Variable(torch.Tensor([sigmoid_bias]))
        self._intermediates_cache = {}
        self.add_inverse_to_cache = True

    def get_arn(self):
        """
        :rtype: pyro.nn.AutoRegressiveNN

        Return the AutoRegressiveNN associated with the InverseAutoregressiveFlow
        """
        return self.arn

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.autograd.Variable

        Invokes the bijection x=>y; in the prototypical context of a TransformedDistribution `x` is a
        sample from the base distribution (or the output of a previous flow)
        """
        hidden = self.arn(x)
        sigma = self.sigmoid(hidden[:, 0:self.input_dim] + self.sigmoid_bias.type_as(hidden))
        mean = hidden[:, self.input_dim:]
        y = sigma * x + (Variable(torch.ones(sigma.size())).type_as(sigma) - sigma) * mean
        self._add_intermediate_to_cache(sigma, y, 'sigma')
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.autograd.Variable

        Inverts y => x. As noted above, this implementation is incapable of inverting arbitrary values
        `y`; rather it assumes `y` is the result of a previously computed application of the bijector
        to some `x` (which was cached on the forward call)
        """
        if (y, 'x') in self._intermediates_cache:
            x = self._intermediates_cache.pop((y, 'x'))
            return x
        else:
            raise KeyError("Bijector InverseAutoregressiveFlow expected to find" +
                           "key in intermediates cache but didn't")

    def _add_intermediate_to_cache(self, intermediate, y, name):
        """
        Internal function used to cache intermediate results computed during the forward call
        """
        assert((y, name) not in self._intermediates_cache),\
            "key collision in _add_intermediate_to_cache"
        self._intermediates_cache[(y, name)] = intermediate

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        if (y, 'sigma') in self._intermediates_cache:
            sigma = self._intermediates_cache.pop((y, 'sigma'))
        else:
            raise KeyError("Bijector InverseAutoregressiveFlow expected to find" +
                           "key in intermediates cache but didn't")
        return torch.log(sigma)
