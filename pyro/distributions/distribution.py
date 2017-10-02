import torch


class Distribution(object):
    """
    Abstract base class for probability distributions.

    Instances can either be constructed from a fixed parameter and called without paramters,
    or constructed without a parameter and called with a paramter.
    It is not allowed to specify a parameter both during construction and when calling.
    When calling with a parameter, it is preferred to use one of the singleton instances
    in pyro.distributions rather than constructing a new instance without a parameter.

    Derived classes must implement the `sample`, and `batch_log_pdf` methods.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for base distribution class.

        Currently takes no explicit arguments.
        """
        self.reparameterized = False

    def __call__(self, *args, **kwargs):
        """
        Samples a random value.

        :return: A random value.
        :rtype: torch.autograd.Variable
        """
        return self.sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Samples a random value.

        :return: A random value.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def log_pdf(self, x, *args, **kwargs):
        """
        Evaluates total log probability density for one or a batch of samples and parameters.

        :param torch.autograd.Variable x: A value.
        :return: total log probability density as a one-dimensional torch.autograd.Variable of size 1.
        :rtype: torch.autograd.Variable
        """
        return torch.sum(self.batch_log_pdf(x, *args, **kwargs))

    def batch_log_pdf(self, x, *args, **kwargs):
        """
        Evaluates log probability densities for one or a batch of samples and parameters.

        :param torch.autograd.Variable x: A single value or a batch of values batched along axis 0.
        :return: log probability densities as a one-dimensional torch.autograd.Variable.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def support(self, *args, **kwargs):
        """
        Returns a representation of the distribution's support.
        :return: A representation of the distribution's support.
        :rtype: torch.Tensor
        """
        raise NotImplementedError("Support not implemented for {}".format(type(self)))
