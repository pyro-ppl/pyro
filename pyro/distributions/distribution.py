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
        pass

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
