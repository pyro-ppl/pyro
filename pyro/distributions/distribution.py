from abc import ABCMeta, abstractmethod

import torch
from six import add_metaclass


@add_metaclass(ABCMeta)
class Distribution(object):
    """
    Base class for parameterized probability distributions.

    Distributions in Pyro are stochastic function objects with `.sample()` and `.log_pdf()` methods.
    Pyro provides two versions of each stochastic function:

    `(i)` lowercase versions that take parameters::

      x = dist.binomial(param)              # Returns a sample of size size(param).
      p = dist.binomial.log_pdf(x, param)   # Evaluates log probability of x.

    and `(ii)` UpperCase distribution classes that can construct stochastic functions with
    fixed parameters::

      d = dist.Binomial(param)
      x = d()                               # Samples a sample of size size(param).
      p = d.log_pdf(x)                      # Evaluates log probability of x.

    **Parameters**:

    Parameters should be of type `torch.autograd.Variable` and all methods return type
    `torch.autograd.Variable` unless otherwise noted.

    **Tensor Shapes**:

    Distributions provide a method `.shape()` for the tensor shape of samples::

      x = d.sample(*args, **kwargs)
      assert x.shape == d.shape(*args, **kwargs)

    Pyro distinguishes two different roles for tensor shapes of samples:

    - The leftmost dimension corresponds to iid *batching*, which can be
      treated specially during inference via the `.batch_log_pdf()` method.
    - The rightmost dimensions correspond to *event shape*.

    These shapes are related by the equation::

      assert d.shape(*args, **kwargs) == (d.batch_shape(*args, **kwargs) +
                                          d.event_shape(*args, **kwargs))

    Distributions provide a vectorized `.batch_log_pdf()` method that evaluates
    the log probability density of each event in a batch independently,
    returning a tensor of shape `d.batch_shape() + (1,)`::

      x = d.sample(*args, **kwargs)
      assert x.shape == d.shape(*args, **kwargs)
      log_p = d.batch_log_pdf(x, *args, **kwargs)
      assert log_p.shape == d.batch_shape(*args, **kwargs) + (1,)

    Distributions may also support broadcasting of the `.log_pdf()` and
    `.batch_log_pdf()` methods, which may each be evaluated with a sample
    tensor `x` that is larger than (but broadcastable from) the parameters.
    In this case, `d.batch_shape()` will return a shape that is broadcastable to
    `d.batch_log_pdf().shape`. For example::

      x = d.sample()
      xx = torch.stack([x, x])
      d.batch_log_pdf(xx) + torch.zeros(d.batch_shape() + (1,))  # Broadcast.

    **Implementing New Distributions**:

    Derived classes must implement the following methods: `.sample()`,
    `.batch_log_pdf()`, `.batch_shape()`, and `.event_shape()`, .
    Discrete classes may also implement the `.enumerate_support()` method to improve
    gradient estimates and set `.enumerable = True`.

    **Examples**:

    Take a look at the examples[link] to see how they interact with inference algorithms.
    """
    reparameterized = False
    enumerable = False

    def __init__(self, reparameterized=None):
        """
        Constructor for base distribution class.

        :param bool reparameterized: Optional argument to override whether
            instance should be considered reparameterized (by default, this
            is decided by the class).
        """
        if reparameterized is not None:
            self.reparameterized = reparameterized

    def batch_shape(self, *args, **kwargs):
        """
        The left-hand tensor shape of samples, used for batching.

        Samples are of shape `d().shape == d.batch_shape() + d.event_shape()`.

        :return: Tensor shape used for batching.
        :rtype: torch.Size
        """
        raise NotImplementedError

    def event_shape(self, *args, **kwargs):
        """
        The right-hand tensor shape of samples, used for individual events.

        Samples are of shape `d().shape == d.batch_shape() + d.event_shape()`.

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

    def shape(self, *args, **kwargs):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape `d().shape == d.batch_shape() + d.event_shape()`.

        :return: Tensor shape of samples.
        :rtype: torch.Size
        """
        return self.batch_shape(*args, **kwargs) + self.event_shape(*args, **kwargs)

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
    def sample(self, *args, **kwargs):
        """
        Samples a random value.

        For tensor distributions, the returned Variable should have the same `.size()` as the
        parameters.

        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be `self.size()`.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def log_pdf(self, x, *args, **kwargs):
        """
        Evaluates total log probability density of a batch of samples.

        :param torch.autograd.Variable x: A value.
        :return: total log probability density as a one-dimensional torch.autograd.Variable of size 1.
        :rtype: torch.autograd.Variable
        """
        return torch.sum(self.batch_log_pdf(x, *args, **kwargs))

    @abstractmethod
    def batch_log_pdf(self, x, *args, **kwargs):
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

    def enumerate_support(self, *args, **kwargs):
        """
        Returns a representation of the parametrized distribution's support.

        This is implemented only by discrete distributions.

        :return: An iterator over the distribution's discrete support.
        :rtype: iterator
        """
        raise NotImplementedError("Support not implemented for {}".format(type(self)))

    def analytic_mean(self, *args, **kwargs):
        """
        Analytic mean of the distribution, to be implemented by derived classes.

        Note that this is optional, and currently only used for testing distributions.

        :return: Analytic mean.
        :rtype: torch.autograd.Variable.
        :raises: NotImplementedError if mean cannot be analytically computed.
        """
        raise NotImplementedError("Method not implemented by the subclass {}".format(type(self)))

    def analytic_var(self, *args, **kwargs):
        """
        Analytic variance of the distribution, to be implemented by derived classes.

        Note that this is optional, and currently only used for testing distributions.

        :return: Analytic variance.
        :rtype: torch.autograd.Variable.
        :raises: NotImplementedError if variance cannot be analytically computed.
        """
        raise NotImplementedError("Method not implemented by the subclass {}".format(type(self)))
