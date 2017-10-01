class Distribution(object):
    """
    Distribution abstract base class
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for base distribution class.

        Currently takes no explicit arguments.
        """
        self.reparameterized = False

    def __call__(self, *args, **kwargs):
        """
        Samples on call
        """
        return self.sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Virtual sample method.
        """
        raise NotImplementedError()

    def log_pdf(self, x):
        raise NotImplementedError()

    def batch_log_pdf(self, x, batch_size):
        raise NotImplementedError()

    def support(self):
        raise NotImplementedError("Support not supported for {}".format(str(type(self))))

    def analytic_mean(self, *args, **kwargs):
        """
        Analytic mean of the distribution, to be implemented by derived classes.
        Note that this is optional, and currently only used for testing distributions.
        :return: Analytic mean, assuming it can be computed analytically given the distribution parameters
        :rtype: torch.autograd.Variable.
        """
        raise NotImplementedError("Method not implemented by the subclass {}".format(str(type(self))))

    def analytic_var(self, *args, **kwargs):
        """
        Analytic variance of the distribution, to be implemented by derived classes.
        Note that this is optional, and currently only used for testing distributions.
        :return: Analytic variance, assuming it can be computed analytically given the distribution parameters
        :rtype: torch.autograd.Variable.
        """
        raise NotImplementedError("Method not implemented by the subclass {}".format(str(type(self))))
