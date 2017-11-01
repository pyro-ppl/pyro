from __future__ import absolute_import, division, print_function

from pyro.distributions import Distribution


class RandomPrimitive(Distribution):
    """
    For help on a RandomPrimitive instance rp, use help(rp.dist_class).
    """
    __slots__ = ['dist_class']

    def __init__(self, dist_class):
        self.dist_class = dist_class
        super(RandomPrimitive, self).__init__()

    @property
    def enumerable(self):
        return self.dist_class.enumerable

    @property
    def reparameterized(self):
        return self.dist_class.reparameterized

    def batch_shape(self, x=None, *args, **kwargs):
        return self.dist_class(*args, **kwargs).batch_shape(x)

    def event_shape(self, *args, **kwargs):
        return self.dist_class(*args, **kwargs).event_shape()

    def event_dim(self, *args, **kwargs):
        return self.dist_class(*args, **kwargs).event_dim()

    def shape(self, x=None, *args, **kwargs):
        return self.dist_class(*args, **kwargs).shape(x)

    def sample(self, *args, **kwargs):
        return self.dist_class(*args, **kwargs).sample()

    __call__ = sample

    def log_pdf(self, x, *args, **kwargs):
        return self.dist_class(*args, **kwargs).log_pdf(x)

    def batch_log_pdf(self, x, *args, **kwargs):
        return self.dist_class(*args, **kwargs).batch_log_pdf(x)

    def enumerate_support(self, *args, **kwargs):
        return self.dist_class(*args, **kwargs).enumerate_support()

    def analytic_mean(self, *args, **kwargs):
        return self.dist_class(*args, **kwargs).analytic_mean()

    def analytic_var(self, *args, **kwargs):
        return self.dist_class(*args, **kwargs).analytic_var()
