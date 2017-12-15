from __future__ import absolute_import, division, print_function

import functools
import os
import warnings

import torch

from pyro.distributions.distribution import Distribution

# TODO Decide based on torch.__version__ once torch.distributions matures.
USE_TORCH_DISTRIBUTIONS = int(os.environ.get('PYRO_USE_TORCH_DISTRIBUTIONS', 0))


def torch_wrapper(pyro_dist):
    """
    Decorator for optional wrappers around torch.distributions classes.
    """
    if not USE_TORCH_DISTRIBUTIONS:
        return lambda wrapper: pyro_dist

    if not hasattr(torch.distributions, pyro_dist.__name__):
        return lambda wrapper: pyro_dist
    torch_dist = getattr(torch.distributions, pyro_dist.__name__)

    def decorator(wrapper):

        @functools.wraps(pyro_dist)
        def wrapper_with_fallback(*args, **kwargs):
            try:
                return wrapper(*args, **kwargs)
            except NotImplementedError as e:
                warnings.warn('{}, falling back to {}'.format(e.message, pyro_dist.__name__), DeprecationWarning)
                return pyro_dist(*args, **kwargs)

        # Set attributes for use by RandomPrimitive.
        wrapper_with_fallback.reparameterized = torch_dist.has_rsample
        wrapper_with_fallback.enumerable = torch_dist.has_enumerate_support
        return wrapper_with_fallback

    return decorator


class TorchDistribution(Distribution):
    """
    Compatibility wrapper around
    `torch.distributions.Distribution <http://pytorch.org/docs/master/_modules/torch/distributions.html#Distribution>`_
    """

    def __init__(self, torch_dist, log_pdf_mask=None, *args, **kwargs):
        super(TorchDistribution, self).__init__(*args, **kwargs)
        self.torch_dist = torch_dist
        self.log_pdf_mask = log_pdf_mask

    def sample(self):
        if self.reparameterized:
            return self.torch_dist.rsample()
        else:
            return self.torch_dist.sample()

    def batch_log_pdf(self, x):
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        log_pxs = self.torch_dist.log_prob(x)
        batch_log_pdf = torch.sum(log_pxs, -1).contiguous().view(batch_log_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        return self.torch_dist.enumerate_support()
