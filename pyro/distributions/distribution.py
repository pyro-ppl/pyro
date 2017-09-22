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
        """
        return self.sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Samples a random value.
        """
        raise NotImplementedError()

    def log_pdf(self, x, *args, **kwargs):
        """
        Evaluates log probability density at a single value.

        :param x: A value.
        :type x: float or int or torch.Tensor
        :return: log probability density
        :rtype: float
        """
        raise NotImplementedError

    def batch_log_pdf(self, xs, *args, **kwargs):
        """
        Evaluates the total log probability of a collection of values.

        :param torch.Tensor xs: A collection of values stacked along axis 0.
        :param int batch_size: Optional size of tensor batches, defaults to 1.
            Must be specified as a keyword argument.
        :return: sum of all log probabilities of collection of values.
        :rtype: float
        """
        batch_size = kwargs.get('batch_size', 1)
        assert xs.dim() >= 1
        assert batch_size > 0
        num_xs = xs.size()[0]
        result = 0.0
        for i in range(num_xs):
            result += self.log_pdf(xs[i], *args, **kwargs)
        return result

    def support(self, *args, **kwargs):
        """
        Returns a representation of the distribution's support.
        :return: A representation of the distribution's support.
        :rtype: torch.Tensor
        """
        raise NotImplementedError("Support not supported for {}".format(str(type(self))))
