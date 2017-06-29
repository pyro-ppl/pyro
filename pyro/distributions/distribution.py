class Distribution(object):
    """
    Distribution abstract base class
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor for base distribution class.

        Currently takes no explicit arguments.
        """
        self.reparametrized = False
        pass

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """
        Virtual sample method.
        """
        raise NotImplementedError()

    def log_pdf(self, x):
        raise NotImplementedError()

    def support(self):
        raise NotImplementedError("Support not supported for {}".format(str(type(self))))
