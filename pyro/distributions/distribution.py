import torch


class Distribution(object):
    """
    Abstract base class for probability distributions.

    Derived classes must implement the `sample`, and `batch_log_pdf` methods.

    Instances can either be constructed from a fixed parameter and called without paramters,
    or constructed without a parameter and called with a paramter.
    It is not allowed to specify a parameter both during construction and when calling.
    When calling with a parameter, it is preferred to use one of the singleton instances
    in pyro.distributions rather than constructing a new instance without a parameter.

    **Tensor Shapes**:

        - The methods `sample`, `log_pdf`, and `batch_log_pdf` often take pytorch.autograd.Variable args.
        - All of these args must agree on their trailing dimension.
        - Any of these args can be batched along an extra dimension 0.
        - If values and parameters are both batched, their batch dimensions must agree.
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

    def analytic_mean(self, *args, **kwargs):
        """
        Analytic mean of the distribution, to be implemented by derived classes.

        Note that this is optional, and currently only used for testing distributions.

        :return: Analytic mean, assuming it can be computed analytically given the distribution parameters
        :rtype: torch.autograd.Variable.
        """
        raise NotImplementedError("Method not implemented by the subclass {}".format(type(self)))

    def analytic_var(self, *args, **kwargs):
        """
        Analytic variance of the distribution, to be implemented by derived classes.

        Note that this is optional, and currently only used for testing distributions.

        :return: Analytic variance, assuming it can be computed analytically given the distribution parameters
        :rtype: torch.autograd.Variable.
        """
        raise NotImplementedError("Method not implemented by the subclass {}".format(type(self)))
