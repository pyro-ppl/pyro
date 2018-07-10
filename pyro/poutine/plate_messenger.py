from __future__ import absolute_import, division, print_function

import numbers
from six.moves import xrange

import torch
from pyro.distributions.distribution import Distribution

from .indep_messenger import IndepMessenger
from .runtime import _DIM_ALLOCATOR, apply_stack


class PlateMessenger(IndepMessenger):
    """
    Allows specifying iarange contexts outside of a model

    Example::

        @plate(name="outer", sites=["x_noise", "xy_noise"], size=320, dim=-1)
        @plate(name="inner", sites=["y_noise", "xy_noise"], size=200, dim=-2)
        def model():
            x_noise = sample("x_noise", dist.Normal(0., 1.).expand_by([320]))
            y_noise = sample("y_noise", dist.Normal(0., 1.).expand_by([200, 1]))
            xy_noise = sample("xy_noise", dist.Normal(0., 1.).expand_by([200, 320]))

    Example::

        x_axis = plate('outer', 320, dim=-1)
        y_axis = plate('inner', 200, dim=-2)
        with x_axis:
            x_noise = sample("x_noise", dist.Normal(loc, scale).expand_by([320]))
        with y_axis:
            y_noise = sample("y_noise", dist.Normal(loc, scale).expand_by([200, 1]))
        with x_axis, y_axis:
            xy_noise = sample("xy_noise", dist.Normal(loc, scale).expand_by([200, 320]))

    """
    def __init__(self, name=None, size=None, dim=None, sites=None):
        super(PlateMessenger, self).__init__(name, size, dim)
        self.sites = sites
        self._installed = False

    # def __iter__(self):
    #     for i in xrange(self.size):
    #         self.next_context()
    #         with self:
    #             yield i if isinstance(i, numbers.Number) else i.item()

    def __exit__(self, *args):
        if self._installed:
            _DIM_ALLOCATOR.free(self.name, self.dim)
            self._installed = False
        self.counter = 0
        return super(PlateMessenger, self).__exit__(*args)

    def _reset(self):
        if self._installed:
            _DIM_ALLOCATOR.free(self.name, self.dim)
        self._installed = False
        self.counter = 0

    def _process_message(self, msg):
        if self.sites is None or msg["name"] in self.sites:
            if not self._installed:
                self.dim = _DIM_ALLOCATOR.allocate(self.name, self.dim)
                self._installed = True
            super(PlateMessenger, self)._process_message(msg)
        elif self.sites is not None and msg["name"] not in self.sites:
            if self._installed:
                _DIM_ALLOCATOR.free(self.name, self.dim)
                self._installed = False


class _Subsample(Distribution):
    """
    Randomly select a subsample of a range of indices.

    Internal use only. This should only be used by `iarange`.
    """

    def __init__(self, size, subsample_size, use_cuda=None):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        :param bool use_cuda: whether to use cuda tensors
        """
        self.size = size
        self.subsample_size = subsample_size
        self.use_cuda = torch.Tensor().is_cuda if use_cuda is None else use_cuda

    def sample(self, sample_shape=torch.Size()):
        """
        :returns: a random subsample of `range(size)`
        :rtype: torch.LongTensor
        """
        if sample_shape:
            raise NotImplementedError
        subsample_size = self.subsample_size
        if subsample_size is None or subsample_size > self.size:
            subsample_size = self.size
        if subsample_size == self.size:
            result = torch.LongTensor(list(range(self.size)))
        else:
            # torch.randperm does not have a CUDA implementation
            result = torch.randperm(self.size, device=torch.device('cpu'))[:self.subsample_size]
        return result.cuda() if self.use_cuda else result

    def log_prob(self, x):
        # This is zero so that iarange can provide an unbiased estimate of
        # the non-subsampled log_prob.
        result = torch.zeros(1)
        return result.cuda() if self.use_cuda else result


class SubsampleMessenger(PlateMessenger):
    """
    Drop-in replacement for irange and iarange
    """

    def __init__(self, name, size=None, subsample_size=None, subsample=None, dim=None, use_cuda=None, sites=None):
        super(SubsampleMessenger, self).__init__(name, size, dim, sites)
        self.subsample_size = subsample_size
        self.subsample = subsample
        self.use_cuda = use_cuda

    def _do_subsample(self, name, size=None, subsample_size=None, subsample=None, use_cuda=None):
        """
        Helper function for subsampling
        """
        if size is None:
            assert subsample_size is None
            assert subsample is None
            size = -1  # This is PyTorch convention for "arbitrary size"
            subsample_size = -1
        elif subsample is None:
            msg = {
                "type": "sample",
                "name": name,
                "fn": _Subsample(size, subsample_size, use_cuda),
                "is_observed": False,
                "args": (),
                "kwargs": {},
                "value": None,
                "infer": {},
                "scale": 1.0,
                "cond_indep_stack": (),
                "done": False,
                "stop": False,
                "continuation": None
            }
            apply_stack(msg)
            subsample = msg["value"]

        if subsample_size is None:
            subsample_size = len(subsample)
        elif subsample is not None and subsample_size != len(subsample):
            raise ValueError("subsample_size does not match len(subsample), {} vs {}.".format(
                subsample_size, len(subsample)) +
                " Did you accidentally use different subsample_size in the model and guide?")

        return size, subsample_size, subsample

    def __enter__(self):
        self.size, self.subsample_size, self.subsample = self._do_subsample(
            self.name, self.size, self.subsample_size, self.subsample, self.use_cuda)
        super(SubsampleMessenger, self).__enter__()
        return self.subsample

    def _process_message(self, msg):
        super(PlateMessenger, self)._process_message(self, msg)
        if self._installed:
            msg["scale"] = (self.size / self.subsample_size) * msg["scale"]
