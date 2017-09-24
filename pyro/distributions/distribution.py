import torch


class Distribution(object):
    """
    Abstract base class for probability distributions.

    Derived classes must implement the `sample`, and `log_pdf` methods.
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
        raise NotImplementedError()

    def log_pdf(self, x, *args, **kwargs):
        """
        Evaluates log probability density at a single value.

        :param torch.autograd.Variable x: A value.
        :return: log probability density as a zero-dimensional torch.autograd.Variable.
        :rtype: torch.autograd.Variable.
        """
        raise NotImplementedError

    def batch_log_pdf(self, xs, *args, **kwargs):
        """
        Evaluates the total log probability of a collection of values.

        :param torch.Tensor xs: A collection of values stacked along axis 0.
        :param int batch_size: Optional size of tensor batches, defaults to 1.
            Must be specified as a keyword argument.
        :param int axis: Optional axis over which xs is batched, defaults to 1.
            Must be specified as a keyword argument.
        :return: sum of all log probabilities of collection of values.
        :rtype: float
        """
        # This is an inefficient reference implementation that aims to be correct and readable.
        # Authors of Distributions should implement efficient versions of this method.
        # Note that batch_size and axis must be specified as kwargs to allow derived classes to
        # add positional args.
        batch_size = kwargs.get('batch_size', 1)
        axis = kwargs.get('axis', 1)  # TODO Change default to 0 or -1, since 1 is weird.
        assert xs.dim() > axis
        assert batch_size > 0
        if axis != 1:
            raise NotImplementedError

        num_values = xs.size()[0]
        result = torch.autograd.Variable(torch.zeros(num_values))
        for i in range(num_values):
            x = torch.index_select(xs, axis, i).squeeze(axis)
            result[i] = self.log_pdf(x, *args, **kwargs)
        return result

    def support(self, *args, **kwargs):
        """
        Returns a representation of the distribution's support.
        :return: A representation of the distribution's support.
        :rtype: torch.Tensor
        """
        raise NotImplementedError("Support not implemented for {}".format(type(self)))
