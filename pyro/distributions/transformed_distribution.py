from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.nn import AutoRegressiveNN


class TransformedDistribution(Distribution):
    """
    Transforms the base distribution by applying a sequence of `Bijector`s to it.
    This results in a scorable distribution (i.e. it has a `log_pdf()` method).

    :param base_distribution: a (continuous) base distribution; samples from this distribution
        are passed through the sequence of `Bijector`s to yield a sample from the
        `TransformedDistribution`
    :type base_distribution: pyro.distribution.Distribution
    :param bijectors: either a single Bijector or a sequence of Bijectors wrapped in a nn.ModuleList
    :returns: the transformed distribution
    """

    def __init__(self, base_distribution, bijectors, *args, **kwargs):
        super(TransformedDistribution, self).__init__(*args, **kwargs)
        self.reparameterized = base_distribution.reparameterized
        self.base_dist = base_distribution
        if isinstance(bijectors, Bijector):
            self.bijectors = nn.ModuleList([bijectors])
        elif isinstance(bijectors, nn.ModuleList):
            for bijector in bijectors:
                assert isinstance(bijector, Bijector), \
                    "bijectors must be a Bijector or a nn.ModuleList of Bijectors"
            self.bijectors = bijectors

    def sample(self, *args, **kwargs):
        """
        :returns: a sample y
        :rtype: torch.autograd.Variable

        Sample from base distribution and pass through bijector(s)
        """
        x = self.base_dist.sample(*args, **kwargs)
        next_input = x
        for bijector in self.bijectors:
            y = bijector(next_input)
            if bijector.add_inverse_to_cache:
                bijector._add_intermediate_to_cache(next_input, y, 'x')
            next_input = y
        return next_input

    def batch_shape(self, x=None, *args, **kwargs):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        return self.base_dist.batch_shape(x, *args, **kwargs)

    def event_shape(self, *args, **kwargs):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        return self.base_dist.batch_shape(*args, **kwargs)

    def log_pdf(self, y, *args, **kwargs):
        """
        :param y: a value sampled from the transformed distribution
        :type y: torch.autograd.Variable

        :returns: the score (the log pdf) of y
        :rtype: torch.autograd.Variable

        Scores the sample by inverting the bijector(s) and computing the score using the score
        of the base distribution and the log det jacobian
        """
        inverses = []
        next_to_invert = y
        for bijector in reversed(self.bijectors):
            inverse = bijector.inverse(next_to_invert)
            inverses.append(inverse)
            next_to_invert = inverse
        log_pdf_base = self.base_dist.log_pdf(inverses[-1], *args, **kwargs)
        log_det_jacobian = self.bijectors[-1].log_det_jacobian(y, *args, **kwargs)
        for bijector, inverse in zip(list(reversed(self.bijectors))[1:], inverses[:-1]):
            log_det_jacobian += bijector.log_det_jacobian(inverse, *args, **kwargs)
        return log_pdf_base - log_det_jacobian

    def batch_log_pdf(self, y, *args, **kwargs):
        raise NotImplementedError("https://github.com/uber/pyro/issues/293")


class Bijector(nn.Module):
    """
    Abstract class `Bijector`. `Bijector` are bijective transformations with computable
    log det jacobians. They are meant for use in `TransformedDistribution`.
    """

    def __init__(self, *args, **kwargs):
        super(Bijector, self).__init__(*args, **kwargs)
        self.add_inverse_to_cache = False

    def __call__(self, *args, **kwargs):
        """
        Virtual forward method

        Invokes the bijection x=>y
        """
        raise NotImplementedError()

    def inverse(self, *args, **kwargs):
        """
        Virtual inverse method

        Inverts the bijection y => x.
        """
        raise NotImplementedError()

    def log_det_jacobian(self, *args, **kwargs):
        """
        Virtual logdet jacobian method.

        Computes the log det jacobian `|dy/dx|`
        """
        raise NotImplementedError()


class InverseAutoregressiveFlow(Bijector):
    """
    An implementation of an Inverse Autoregressive Flow. Together with the `TransformedDistribution` this
    provides a way to create richer variational approximations.

    Example usage::

    >>> base_dist = Normal(...)
    >>> iaf = InverseAutoregressiveFlow(...)
    >>> pyro.module("my_iaf", iaf)
    >>> iaf_dist = TransformedDistribution(base_dist, iaf)

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

    def __call__(self, x, *args, **kwargs):
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

    def inverse(self, y, *args, **kwargs):
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

    def log_det_jacobian(self, y, *args, **kwargs):
        """
        Calculates the determinant of the log jacobian
        """
        if (y, 'sigma') in self._intermediates_cache:
            sigma = self._intermediates_cache.pop((y, 'sigma'))
        else:
            raise KeyError("Bijector InverseAutoregressiveFlow expected to find" +
                           "key in intermediates cache but didn't")
        if 'log_pdf_mask' in kwargs:
            return torch.sum(kwargs['log_pdf_mask'] * torch.log(sigma))
        return torch.sum(torch.log(sigma))
