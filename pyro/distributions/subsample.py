import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.util import log_gamma


class Subsample(Distribution):
    """
    Randomly select a subsample of a range of indices.

    :param int size: the size of the range to subsample from
    :param int subsample_size: the size of the returned subsample
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
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        """
        self.size = size
        self.subsample_size = subsample_size

    def sample(self, size=None, subsample_size=None):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        :returns: a random subsample of `range(size)`
        :rtype: torch.autograd.Variable of torch.LongTensor
        """
        size, subsample_size = self._sanitize_inputs(size, subsample_size)
        if subsample_size > size:
            raise ValueError("expected size < subsample_size, actual {} vs {}".format(
                size, subsample_size))
        return Variable(torch.randperm(size)[:subsample_size])

    def batch_log_pdf(self, x, size=None, subsample_size=None):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        :param torch.autograd.Variable x: a subsample of `range(size)`
        :return: log probability density
        :rtype: torch.autograd.Variable
        """
        size, subsample_size = self._sanitize_inputs(size, subsample_size)
        if subsample_size is None:
            subsample_size = x.size(-1)
        elif subsample_size != x.size(-1):
            raise ValueError("subsample does not match subsample_size")
        return log_gamma(1 + size) - log_gamma(1 + size - subsample_size)
