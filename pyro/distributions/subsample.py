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
    reparameterized = False

    def _sanitize_inputs(self, size, subsample_size, use_cuda):
        if size is None:
            size = self.size
        if subsample_size is None:
            subsample_size = self.subsample_size
        if use_cuda is None:
            use_cuda = self.use_cuda
        return size, subsample_size, use_cuda

    def __init__(self, size=None, subsample_size=None, use_cuda=False):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        """
        self.size = size
        self.subsample_size = subsample_size
        self.use_cuda = use_cuda

    def sample(self, size=None, subsample_size=None, use_cuda=None):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        :returns: a random subsample of `range(size)`
        :rtype: torch.autograd.Variable of torch.LongTensor
        """
        size, subsample_size, use_cuda = self._sanitize_inputs(size, subsample_size, use_cuda)
        if subsample_size > size:
            raise ValueError("expected size < subsample_size, actual {} vs {}".format(
                size, subsample_size))
        ix = Variable(torch.randperm(size)[:subsample_size])
        if use_cuda:
            ix = ix.cuda()
        return ix

    def batch_log_pdf(self, x, size=None, subsample_size=None, use_cuda=None):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        :param torch.autograd.Variable x: a subsample of `range(size)`
        :return: log probability density
        :rtype: torch.autograd.Variable
        """
        size, subsample_size, use_cuda = self._sanitize_inputs(size, subsample_size, use_cuda)
        if subsample_size is None:
            subsample_size = x.size(-1)
        elif subsample_size != x.size(-1):
            raise ValueError("subsample does not match subsample_size")
        size = Variable(torch.Tensor([size]))
        ret = log_gamma(1 + size) - log_gamma(1 + size - subsample_size)
        if use_cuda:
            ret = ret.cuda()
        return ret
