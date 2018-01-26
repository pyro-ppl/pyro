from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions import Distribution


class RandomPrimitive(Distribution):
    """
    For help on a RandomPrimitive instance rp, use help(rp.dist_class).
    """
    __slots__ = ['dist_class']

    def __init__(self, dist_class):
        if dist_class.stateful:
            raise TypeError('Cannot wrap stateful class {} in RandomPrimitive.'.format(type(dist_class)))
        self.dist_class = dist_class
        super(RandomPrimitive, self).__init__()

    @property
    def enumerable(self):
        return self.dist_class.enumerable

    @property
    def reparameterized(self):
        return self.dist_class.reparameterized

    def batch_shape(self, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).batch_shape()

    def event_shape(self, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).event_shape()

    def event_dim(self, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).event_dim()

    def shape(self, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', torch.Size())
        return self.dist_class(*args, **kwargs).shape(sample_shape)

    def sample(self, *args, **kwargs):
        sample_shape = kwargs.pop('sample_shape', torch.Size())
        return self.dist_class(*args, **kwargs).sample(sample_shape)

    __call__ = sample

    def log_prob(self, x, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).log_prob(x)

    def score_parts(self, x, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).score_parts(x)

    def enumerate_support(self, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).enumerate_support()

    def analytic_mean(self, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).analytic_mean()

    def analytic_var(self, *args, **kwargs):
        kwargs.pop('sample_shape', None)
        return self.dist_class(*args, **kwargs).analytic_var()
