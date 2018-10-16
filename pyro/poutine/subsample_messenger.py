from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.distribution import Distribution

from .indep_messenger import CondIndepStackFrame, IndepMessenger
from .runtime import apply_stack


class _Subsample(Distribution):
    """
    Randomly select a subsample of a range of indices.

    Internal use only. This should only be used by `iarange`.
    """

    def __init__(self, size, subsample_size, use_cuda=None, device=None):
        """
        :param int size: the size of the range to subsample from
        :param int subsample_size: the size of the returned subsample
        :param bool use_cuda: DEPRECATED, use the `device` arg instead.
            Whether to use cuda tensors.
        :param str device: device to place the `sample` and `log_prob`
            results on.
        """
        self.size = size
        self.subsample_size = subsample_size
        self.use_cuda = use_cuda
        if self.use_cuda is not None:
            if self.use_cuda ^ (device != "cpu"):
                raise ValueError("Incompatible arg values use_cuda={}, device={}."
                                 .format(use_cuda, device))
        self.device = torch.Tensor().device if not device else device

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
        if subsample_size >= self.size:
            result = torch.arange(self.size, dtype=torch.long).to(self.device)
        else:
            result = torch.multinomial(torch.ones(self.size), self.subsample_size,
                                       replacement=False).to(self.device)
        return result.cuda() if self.use_cuda else result

    def log_prob(self, x):
        # This is zero so that iarange can provide an unbiased estimate of
        # the non-subsampled log_prob.
        result = torch.tensor(0., device=self.device)
        return result.cuda() if self.use_cuda else result


class SubsampleMessenger(IndepMessenger):
    """
    Drop-in replacement for irange and iarange, including subsampling!
    """

    def __init__(self, name, size=None, subsample_size=None, subsample=None, dim=None,
                 use_cuda=None, device=None):
        super(SubsampleMessenger, self).__init__(name, size, dim)
        self.subsample_size = subsample_size
        self._indices = subsample
        self.use_cuda = use_cuda
        self.device = device

        self.size, self.subsample_size, self._indices = self._subsample(
            self.name, self.size, self.subsample_size,
            self._indices, self.use_cuda, self.device)

    @staticmethod
    def _subsample(name, size=None, subsample_size=None, subsample=None, use_cuda=None, device=None):
        """
        Helper function for iarange and irange. See their docstrings for details.
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
                "fn": _Subsample(size, subsample_size, use_cuda, device),
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

    def _reset(self):
        self.subsample = None
        super(SubsampleMessenger, self)._reset()

    def _process_message(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size, self.counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        msg["scale"] = (self.size / self.subsample_size) * msg["scale"]
