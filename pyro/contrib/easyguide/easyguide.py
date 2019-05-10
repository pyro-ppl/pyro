from __future__ import absolute_import, division, print_function

import functools
import re

from torch.distributions import biject_to

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import sum_rightmost
from pyro.poutine.util import prune_subsample_sites

try:
    from contextlib import ExitStack  # python 3
except ImportError:
    from contextlib2 import ExitStack  # python 2


def _product(shape):
    """
    Computes the product of the dimensions of a given shape tensor
    """
    result = 1
    for size in shape:
        result *= size
    return result


def _make_matcher(match):
    if isinstance(match, str):
        return re.compile(match).search
    if isinstance(match, (list, set)):
        return match.__contains__
    return match


class EasyGuide(object):
    def __init__(self, model, guide_fn=None, prefix="easy"):
        self.model = model
        self.prefix = prefix
        self.prototype_trace = None
        if guide_fn is not None:
            self.guide_fn = guide_fn

    def guide_fn(self, *args, **kwargs):
        raise NotImplementedError

    def _setup_prototype(self, *args, **kwargs):
        with poutine.block():
            self.prototype_trace = poutine.trace(self.model).get_trace(*args, **kwargs)
        self.prototype_trace = prune_subsample_sites(self.prototype_trace)

        self._cond_indep_frames = {}
        self._cond_indep_stacks = {}
        self._unconstrained_shapes = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            # Collect plates.
            for frame in site["cond_indep_stack"]:
                if frame.vectorized:
                    self._cond_indep_frames[frame.name] = frame
                else:
                    raise NotImplementedError("EasyGuide does not support sequential pyro.plate")

            # Collect independence contexts.
            self._cond_indep_stacks[name] = site["cond_indep_stack"]

            # Collect the shapes of unconstrained values.
            # These may differ from the shapes of constrained values.
            self._unconstrained_shapes[name] = biject_to(site["fn"].support).inv(site["value"]).shape

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(self, *args, **kwargs)
        self._plates = {frame.name: pyro.plate(frame.name, frame.size, dim=frame.dim)
                        for frame in sorted(self._cond_indep_frames.values())}
        return self.guide_fn(*args, **kwargs)

    def sample(self, aux_name, easy_dist, match):
        aux_name = "{}_{}".format(self.prefix, aux_name)
        match = _make_matcher(match)
        names = [n for n in self.prototype_trace.nodes if match(n)]
        blob_dim = sum(_product(self._unconstrained_shapes[n]) for n in names)
        blob = pyro.sample(aux_name, easy_dist(blob_dim), infer={"is_auxiliary": True})

        values = {}
        pos = 0
        for name in names:
            site = self.prototype_trace.nodes[name]
            unconstrained_shape = self._unconstrained_shapes[name]
            size = _product(unconstrained_shape)
            unconstrained_value = blob[pos:pos + size].reshape(unconstrained_shape)
            pos += size
            transform = biject_to(site["fn"].support)
            value = transform(unconstrained_value)
            log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
            log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + site["fn"].event_dim)
            delta_dist = dist.Delta(value, log_density=log_density, event_dim=site["fn"].event_dim)

            with ExitStack() as stack:
                for frame in self._cond_indep_stacks[name]:
                    stack.enter_context(self._plates[frame.name])
                values[name] = pyro.sample(name, delta_dist)

        return blob, values


def easyguide(model, **kwargs):
    assert "guide_fn" not in kwargs
    return functools.partial(EasyGuide, model, **kwargs)
