# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.distributions.distribution import Distribution
from pyro.poutine.util import is_validation_enabled
from pyro.util import ignore_jit_warnings

from .indep_messenger import CondIndepStackFrame, IndepMessenger
from .runtime import apply_stack


class _Subsample(Distribution):
    """
    Randomly select a subsample of a range of indices.

    Internal use only. This should only be used by `plate`.
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
        with ignore_jit_warnings(["torch.Tensor results are registered as constants"]):
            self.device = torch.Tensor().device if not device else device

    @ignore_jit_warnings(["Converting a tensor to a Python boolean"])
    def sample(self, sample_shape=torch.Size()):
        """
        :returns: a random subsample of `range(size)`
        :rtype: torch.LongTensor
        """
        if sample_shape:
            raise NotImplementedError
        subsample_size = self.subsample_size
        if subsample_size is None or subsample_size >= self.size:
            result = torch.arange(self.size, device=self.device)
        else:
            result = torch.randperm(self.size, device=self.device)[:subsample_size].clone()
        return result.cuda() if self.use_cuda else result

    def log_prob(self, x):
        # This is zero so that plate can provide an unbiased estimate of
        # the non-subsampled log_prob.
        result = torch.tensor(0., device=self.device)
        return result.cuda() if self.use_cuda else result


class SubsampleMessenger(IndepMessenger):
    """
    Extension of IndepMessenger that includes subsampling.
    """

    def __init__(self, name, size=None, subsample_size=None, subsample=None, dim=None,
                 use_cuda=None, device=None):
        super().__init__(name, size, dim, device)
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
        Helper function for plate. See its docstrings for details.
        """
        if size is None:
            assert subsample_size is None
            assert subsample is None
            size = -1  # This is PyTorch convention for "arbitrary size"
            subsample_size = -1
        else:
            msg = {
                "type": "sample",
                "name": name,
                "fn": _Subsample(size, subsample_size, use_cuda, device),
                "is_observed": False,
                "args": (),
                "kwargs": {},
                "value": subsample,
                "infer": {},
                "scale": 1.0,
                "mask": None,
                "cond_indep_stack": (),
                "done": False,
                "stop": False,
                "continuation": None
            }
            apply_stack(msg)
            subsample = msg["value"]

        with ignore_jit_warnings():
            if subsample_size is None:
                subsample_size = subsample.size(0) if isinstance(subsample, torch.Tensor) \
                    else len(subsample)
            elif subsample is not None and subsample_size != len(subsample):
                raise ValueError("subsample_size does not match len(subsample), {} vs {}.".format(
                    subsample_size, len(subsample)) +
                    " Did you accidentally use different subsample_size in the model and guide?")

        return size, subsample_size, subsample

    def _reset(self):
        self._indices = None
        super()._reset()

    def _process_message(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.subsample_size, self.counter)
        frame.full_size = self.size  # Used for param initialization.
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        if isinstance(self.size, torch.Tensor) or isinstance(self.subsample_size, torch.Tensor):
            if not isinstance(msg["scale"], torch.Tensor):
                with ignore_jit_warnings():
                    msg["scale"] = torch.tensor(msg["scale"])
        msg["scale"] = msg["scale"] * self.size / self.subsample_size

    def _postprocess_message(self, msg):
        if msg["type"] in ("param", "subsample") and self.dim is not None:
            event_dim = msg["kwargs"].get("event_dim")
            if event_dim is not None:
                assert event_dim >= 0
                dim = self.dim - event_dim
                shape = msg["value"].shape
                if len(shape) >= -dim and shape[dim] != 1:
                    if is_validation_enabled() and shape[dim] != self.size:
                        if msg["type"] == "param":
                            statement = "pyro.param({}, ..., event_dim={})".format(msg["name"], event_dim)
                        else:
                            statement = "pyro.subsample(..., event_dim={})".format(event_dim)
                        raise ValueError(
                            "Inside pyro.plate({}, {}, dim={}) invalid shape of {}: {}"
                            .format(self.name, self.size, self.dim, statement, shape))
                    # Subsample parameters with known batch semantics.
                    if self.subsample_size < self.size:
                        value = msg["value"]
                        new_value = value.index_select(dim, self._indices)
                        if msg["type"] == "param":
                            if hasattr(value, "_pyro_unconstrained_param"):
                                param = value._pyro_unconstrained_param
                            else:
                                param = value.unconstrained()

                            if not hasattr(param, "_pyro_subsample"):
                                param._pyro_subsample = {}

                            param._pyro_subsample[dim] = self._indices
                            new_value._pyro_unconstrained_param = param
                        msg["value"] = new_value
