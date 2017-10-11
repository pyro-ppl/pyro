import torch
from torch.autograd import Variable
from scipy.special import gammaln

from pyro.distributions.distribution import Distribution


class Subsample(Distribution):
    """
    Randomly select a subsample of a range of indices.

    :param int size: the size of the range to subsample from
    :param int batch_size: the size of the returned subsample
    :returns: a random subsample of `range(size)`
    :rtype: torch.autograd.Variable of torch.LongTensor
    """

    def _sanitize_inputs(self, size, subsample_size):
        if size is None:
            size = self.size
        if subsample_size is None:
            subsample_size = self.subsample_size
        return size, subsample_size

    def __init__(self, size=None, subsample_size=None):
        self.size = size
        self.subsample_size = subsample_size

    def sample(self, size=None, subsample_size=None):
        size, subsample_size = self._sanitize_inputs(size, subsample_size)
        return Variable(torch.randperm(size)[:subsample_size])

    def batch_log_pdf(self, x, size=None, subsample_size=None):
        size, subsample_size = self._sanitize_inputs(size, subsample_size)
        if subsample_size is None:
            subsample_size = x.size(-1)
        elif subsample_size != x.size(-1):
            raise ValueError('subsample does not match subsample_size')
        return gammaln(1 + size) - gammaln(1 + size - subsample_size)
