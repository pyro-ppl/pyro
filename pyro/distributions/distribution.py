import torch


class Distribution(object):
    """
    Base class for parametrized probability distributions.

    Distributions in Pyro are stochastic function objects with `.sample()` and `.log_pdf()` methods.
    Pyro provides two versions of each stochastic function lowercase versions that take parameters::

      x = dist.binomial(param)              # Returns a sample of size size(param).
      p = dist.binomial.log_pdf(x, param)   # Evaluates log probability of x.

    as well as UpperCase distribution classes that can construct stochastic functions with
    fixed parameters::

      d = dist.Binomial(param)
      x = d()                               # Samples a sample of size size(param).
      p = d.log_pdf(x)                      # Evaluates log probability of x.

    **Parameters**:

    Parameters should be of type `torch.autograd.Variable` and all methods return type
    `torch.autograd.Variable` unless otherwise noted.

    **Tensor Shapes**:

    Pyro distinguishes two different roles for tensor dimensions:

    - The leftmost dimension corresponds to *batching*, which is treated
      specially during inference.
    - The rightmost dimensions correspond to *event shape*.

    These two shapes are related by the equation::

      d.shape == d.batch_shape + d.event_shape

    Distributions also provide a vectorized `.batch_log_pdf()` method that
    evaluates the log probability density of each event in a batch
    independently, returning a tensor of shape `d.batch_shape`.

    **Implementing New Distributions**:

    Derived classes must implement the following methods: `.batch_shape`,
    `.event_shape`, `.sample()`, and `.batch_log_pdf()`.
    Discrete classes should also implement the `.support()` method to imporove
    gradient estimates.

    **Examples**:

    Take a look at the examples[link] to see how they interact with inference algorithms.
    """

    enumerable = False

    def __init__(self, *args, **kwargs):
        """
        Constructor for base distribution class.

        Currently takes no explicit arguments.
        """
        self.reparameterized = False

    @property
    def batch_shape(self):
        """
        The left-hand tensor size of samples from this distribution, used for batching.

        :return: Tensor shape used for batching.
        :rtype: torch.Size
        """
        raise NotImplementedError

    @property
    def event_shape(self):
        """
        The right-hand tensor size of this distribution, used for individual events.

        :return: Tensor shape used for individual events.
        :rtype: torch.Size
        """
        raise NotImplementedError

    @property
    def shape(self):
        """
        The size of batch events from this distribution.

        :return: Tensor shape used for individual events.
        :rtype: torch.Size
        """
        return self.batch_shape + self.event_shape

    def __call__(self, *args, **kwargs):
        """
        Samples a random value (just an alias for `.sample(*args, **kwargs)`).

        For tensor distributions, the returned Variable should have the same `.size()` as the
        parameters.

        :return: A random value.
        :rtype: torch.autograd.Variable
        """
        return self.sample(*args, **kwargs)

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

    def support(self, *args, **kwargs):
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
