from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod

from six import add_metaclass

import torch
from pyro.distributions.score_parts import ScoreParts


@add_metaclass(ABCMeta)
class Distribution(object):
    """
    Base class for parameterized probability distributions.

    Distributions in Pyro are stochastic function objects with ``.sample()`` and
    ``.log_prob()`` methods. Pyro provides two versions of each stochastic function:

    `(i)` lowercase versions that take parameters::

      x = dist.bernoulli(param)              # Returns a sample of size size(param).
      p = dist.bernoulli.log_prob(x, param)  # Evaluates log probability of x.

    and `(ii)` UpperCase distribution classes that can construct stochastic functions with
    fixed parameters::

      d = dist.Bernoulli(param)
      x = d()                                # Samples a sample of size size(param).
      p = d.log_prob(x)                      # Evaluates log probability of x.

    Under the hood the lowercase versions are aliases for the UpperCase versions.

    .. note::

        Parameters and data should be of type `torch.autograd.Variable` and all
        methods return type `torch.autograd.Variable` unless otherwise noted.

    **Tensor Shapes**:

    Distributions provide a method ``.shape()`` for the tensor shape of samples::

      x = d.sample(*args, **kwargs)
      assert x.shape == d.shape(*args, **kwargs)

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
    returning a tensor of shape ``d.batch_shape(x)``::

      x = d.sample(*args, **kwargs)
      assert x.size() == d.shape(sample_shape, *args, **kwargs)
      log_p = d.log_prob(x, *args, **kwargs)
      assert log_p.size() == d.batch_shape(*args, **kwargs)

    **Implementing New Distributions**:

    Derived classes must implement the following methods: ``.sample()``,
    ``.log_prob()``, ``.batch_shape()``, and ``.event_shape()``.
    Discrete classes may also implement the ``.enumerate_support()`` method to improve
    gradient estimates and set ``.enumerable = True``.

    **Examples**:

    Take a look at the `examples <http://pyro.ai/examples>`_ to see how they interact
    with inference algorithms.
    """
    stateful = False
    reparameterized = False
    enumerable = False

    def batch_shape(self, *args, **kwargs):
        """
        The left-hand tensor shape of parameters used to index the (possibly)
        non-identical independent draws from the distribution.

        Samples are of shape :code:`d.shape(sample_shape) == sample_shape +
        d.batch_shape() + d.event_shape()`.

        :return: Tensor shape used for batching.
        :rtype: torch.Size
        :raises: ValueError if the parameters are not broadcastable to the data shape
        """
        raise NotImplementedError

    def event_shape(self, *args, **kwargs):
        """
        The right-hand tensor shape of parameters used for individual events. The
        event dimension(/s) is used to designate random variables that could
        potentially depend on each other, for instance in the case of Dirichlet
        or the OneHotCategorical distribution. Note that ``event_shape`` is
        empty for all univariate distributions.

        Samples are of shape :code:`d.shape(sample_shape) == sample_shape +
        d.batch_shape() + d.event_shape()`.

        :return: Tensor shape used for individual events.
        :rtype: torch.Size
        """
        raise NotImplementedError

    def event_dim(self, *args, **kwargs):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape(*args, **kwargs))

    def shape(self, sample_shape=torch.Size(), *args, **kwargs):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape :code:`d.shape(sample_shape) == sample_shape +
        d.batch_shape() + d.event_shape()`.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: Tensor shape of samples.
        :rtype: torch.Size
        """
        return sample_shape + self.batch_shape(*args, **kwargs) + self.event_shape(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Samples a random value (just an alias for `.sample(*args, **kwargs)`).

        For tensor distributions, the returned Variable should have the same `.size()` as the
        parameters.

        :return: A random value.
        :rtype: torch.autograd.Variable
        """
        return self.sample(*args, **kwargs)

    @abstractmethod
    def sample(self, sample_shape=torch.Size(), *args, **kwargs):
        """
        Samples a random value.

        For tensor distributions, the returned Variable should have the same `.size()` as the
        parameters, unless otherwise noted.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be `self.size()`.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x, *args, **kwargs):
        """
        Evaluates log probability densities for each of a batch of samples.

        :param torch.autograd.Variable x: A single value or a batch of values
            batched along axis 0.
        :return: log probability densities as a one-dimensional
            `torch.autograd.Variable` with same batch size as value and params.
            The shape of the result should be `self.batch_size()`.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def score_parts(self, x, *args, **kwargs):
        """
        Computes ingredients for stochastic gradient estimators of ELBO.

        The default implementation is correct both for non-reparameterized and
        for fully reparameterized distributions. Partially reparameterized
        distributions should override this method to compute correct
        `.score_function` and `.entropy_term` parts.

        :param torch.autograd.Variable x: A single value or batch of values.
        :return: A `ScoreParts` object containing parts of the ELBO estimator.
        :rtype: ScoreParts
        """
        log_pdf = self.log_prob(x, *args, **kwargs)
        if self.reparameterized:
            return ScoreParts(log_pdf=log_pdf, score_function=0, entropy_term=log_pdf)
        else:
            # XXX should the user be able to control inclusion of the entropy term?
            # See Roeder, Wu, Duvenaud (2017) "Sticking the Landing" https://arxiv.org/abs/1703.09194
            return ScoreParts(log_pdf=log_pdf, score_function=log_pdf, entropy_term=0)

    def enumerate_support(self, *args, **kwargs):
        """
        Returns a representation of the parametrized distribution's support,
        along the first dimension. This is implemented only by discrete
        distributions.

        Note that this returns support values of all the batched RVs in
        lock-step, rather than the full cartesian product.

        :return: An iterator over the distribution's discrete support.
        :rtype: iterator
        """
        raise NotImplementedError("Support not implemented for {}".format(type(self)))
