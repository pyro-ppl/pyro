from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.util import sum_rightmost


class TorchDistributionMixin(Distribution):
    """
    Mixin to provide Pyro compatibility for PyTorch distributions.

    This is mainly useful for wrapping existing PyTorch distributions for
    use in Pyro.  Derived classes must first inherit from
    :class:`torch.distributions.Distribution` and then inherit from
    :class:`TorchDistributionMixin`.
    """
    @property
    def reparameterized(self):
        return self.has_rsample

    @property
    def enumerable(self):
        return self.has_enumerate_support

    def __call__(self, sample_shape=torch.Size()):
        """
        Samples a random value. This is reparameterized whenever possible.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be `self.shape()`.
        :rtype: torch.autograd.Variable
        """
        return self.rsample(sample_shape) if self.has_rsample else self.sample(sample_shape)

    @property
    def event_dim(self):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape)

    def shape(self, sample_shape=torch.Size()):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape::

          d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: Tensor shape of samples.
        :rtype: torch.Size
        """
        return sample_shape + self.batch_shape + self.event_shape

    def reshape(self, sample_shape=torch.Size(), extra_event_dims=0):
        """
        Reshapes a distribution by adding ``sample_shape`` to its total shape
        and adding ``extra_event_dims`` to its ``event_shape``.

        :param torch.Size sample_shape: The size of the iid batch to be drawn
            from the distribution.
        :param int extra_event_dims: The number of extra event dimensions that
            will be considered dependent.
        :return: A reshaped copy of this distribution.
        :rtype: :class:`Reshape`
        """
        return Reshape(self, sample_shape, extra_event_dims)

    def analytic_mean(self):
        return self.mean

    def analytic_var(self):
        return self.variance


class TorchDistribution(torch.distributions.Distribution, TorchDistributionMixin):
    """
    Base class for PyTorch-compatible distributions with Pyro support.

    This should be the base class for almost all new Pyro distributions.

    .. note::

        Parameters and data should be of type `torch.autograd.Variable` and all
        methods return type `torch.autograd.Variable` unless otherwise noted.

    **Tensor Shapes**:

    TorchDistributions provide a method ``.shape()`` for the tensor shape of samples::

      x = d.sample(sample_shape)
      assert x.shape == d.shape(sample_shape)

    Pyro follows the same distribution shape semantics as PyTorch. It distinguishes
    between three different roles for tensor shapes of samples:

    - *sample shape* corresponds to the shape of the iid samples drawn from the distribution.
      This is taken as an argument by the distribution's `sample` method.
    - *batch shape* corresponds to non-identical (independent) parameterizations of
      the distribution, inferred from the distribution's parameter shapes. This is
      fixed for a distribution instance.
    - *event shape* corresponds to the event dimensions of the distribution, which
      is fixed for a distribution class. These are collapsed when we try to score
      a sample from the distribution via `d.log_prob(x)`.

    These shapes are related by the equation::

      assert d.shape(sample_shape, *args, **kwargs) == sample_shape +
                                                       d.batch_shape(*args, **kwargs) +
                                                       d.event_shape(*args, **kwargs)

    Distributions provide a vectorized ``.log_prob()`` method that evaluates
    the log probability density of each event in a batch independently,
    returning a tensor of shape ``sample_shape + d.batch_shape``::

      x = d.sample(sample_shape)
      assert x.size() == d.shape(sample_shape)
      log_p = d.log_prob(x)
      assert log_p.shape == sample_shape + d.batch_shape

    **Implementing New Distributions**:

    Derived classes must implement the following methods: ``.rsample()``
    (or ``.sample()`` if ``.has_rsample == True``),
    ``.log_prob()``, ``.batch_shape``, and ``.event_shape``.
    Discrete classes may also implement the ``.enumerate_support()`` method to improve
    gradient estimates and set ``.has_enumerate_support = True``.
    """
    pass


class Reshape(TorchDistribution):
    """
    Reshapes a distribution by adding ``sample_shape`` to its total shape
    and adding ``extra_event_dims`` to its ``event_shape``.

    :param torch.Size sample_shape: The size of the iid batch to be drawn from
        the distribution.
    :param int extra_event_dims: The number of extra event dimensions that will
        be considered dependent.
    """
    def __init__(self, base_dist, sample_shape=torch.Size(), extra_event_dims=0):
        sample_shape = torch.Size(sample_shape)
        self.base_dist = base_dist
        self.sample_shape = sample_shape
        self.extra_event_dims = extra_event_dims
        shape = sample_shape + base_dist.batch_shape + base_dist.event_shape
        batch_dim = len(shape) - extra_event_dims - len(base_dist.event_shape)
        batch_shape, event_shape = shape[:batch_dim], shape[batch_dim:]
        super(Reshape, self).__init__(batch_shape, event_shape)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(self.sample_shape + sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(self.sample_shape + sample_shape)

    def log_prob(self, value):
        return sum_rightmost(self.base_dist.log_prob(value), self.extra_event_dims)

    def score_parts(self, value):
        log_pdf, score_function, entropy_term = self.base_dist.score_parts(value)
        log_pdf = sum_rightmost(log_pdf, self.extra_event_dims)
        score_function = sum_rightmost(score_function, self.extra_event_dims)
        entropy_term = sum_rightmost(entropy_term, self.extra_event_dims)
        return ScoreParts(log_pdf, score_function, entropy_term)

    def enumerate_support(self):
        samples = self.base_dist.enumerate_support()
        if not self.sample_shape:
            return samples
        enum_shape, base_shape = samples.shape[:1], samples.shape[1:]
        samples = samples.contiguous()
        samples = samples.view(enum_shape + (1,) * len(self.sample_shape) + base_shape)
        samples = samples.expand(enum_shape + self.sample_shape + base_shape)
        return samples

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance
