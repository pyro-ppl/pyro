# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide.guides import AutoGuide, _deep_getattr, _deep_setattr
from pyro.infer.autoguide.initialization import InitMessenger, init_to_feasible
from pyro.nn.module import PyroModule, PyroParam
from pyro.ops.tensor_utils import dct, idct

_EVEN_SIZES = {}


def _even_size(size):
    """
    EXPERIMENTAL Returns the next largest number ``n >= size`` whose prime
    factors are all 2, 3, or 5. These sizes are efficient for fast fourier
    transforms.

    :param int size: A positive number.
    :returns: A possibly larger number.
    :rtype int:
    """
    assert isinstance(size, int) and size > 0
    key = size
    if key in _EVEN_SIZES:
        return _EVEN_SIZES[key]
    while True:
        remaining = size
        for n in (2, 3, 5):
            while remaining % n == 0:
                remaining //= n
        if remaining == 1:
            _EVEN_SIZES[key] = size
            return size
        size += 1


class AutoTemporal(AutoGuide):
    """
    Automatic guide that uses a diagonal normal distributions for time-global
    variables and structured normal distributions for time-local variables. The
    structured normal has covariance::

        cov = time_diag + idct @ freq_diag @ dct

    where ``time_diag`` and ``freq_diag`` are both diagonal matrices, and
    :func:`~pyro.ops.tensor_utils.idct` and :func:`~pyro.ops.tensor_utils.dct`
    are orthonormal discrete cosine transforms.

    :param callable model: A generative model.
    :param callable init_loc_fn: A per-site initialization function.
        See :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial scale for the standard deviation of each
        (unconstrained transformed) latent variable.
    """
    def __init__(self, model, init_loc_fn=init_to_feasible, init_scale=0.1):
        self.init_loc_fn = init_loc_fn

        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError("Expected init_scale > 0. but got {}".format(init_scale))
        self._init_scale = init_scale

        model = InitMessenger(self.init_loc_fn)(model)
        super().__init__(model)

    def _create_plates(self):
        result = super()._create_plates()
        if "time" in result:
            # Add a frequency plate matching the time plate.
            result["frequency"] = pyro.plate("frequency", result["time"].size, dim=-1)
        return result

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)

        self._event_dims = {}
        self._is_temporal = {}
        self.locs = PyroModule()
        self.scales = PyroModule()
        self.f_locs = PyroModule()
        self.f_scales = PyroModule()

        # Initialize guide params.
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with torch.no_grad():
                loc = biject_to(site["fn"].support).inv(site["value"].detach())
                scale = torch.full_like(loc, self._init_scale)
            event_dim = site["fn"].event_dim + loc.dim() - site["value"].dim()
            self._event_dims[name] = event_dim
            self._is_temporal[name] = any(f.name == "time" for f in site["cond_indep_stack"])
            if self._is_temporal[name]:
                # Split parameters into time and frequency part.
                loc *= 0.5
                scale *= 0.5 ** 0.5
                f_loc = dct(loc, dim=-1 - event_dim)
                f_scale = scale.clone()

                _deep_setattr(self.f_locs, name, PyroParam(f_loc, constraints.real, event_dim))
                _deep_setattr(self.f_scales, name, PyroParam(f_scale, constraints.positive, event_dim))
            _deep_setattr(self.locs, name, PyroParam(loc, constraints.real, event_dim))
            _deep_setattr(self.scales, name, PyroParam(scale, constraints.positive, event_dim))

    def forward(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            event_dim = self._unconstrained_event_dims[name]
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized and frame.name != "time":
                        stack.enter_context(plates[frame.name])

                loc = _deep_getattr(self.locs, name)
                scale = _deep_getattr(self.scales, name)

                # For temporal sites, add uncertainty in the frequency domain.
                if self._is_temporal[name]:
                    with plates["frequency"]:
                        f_loc = _deep_getattr(self.f_locs, name)
                        f_scale = _deep_getattr(self.f_scales, name)
                        f = pyro.sample(name + "_frequency",
                                        dist.Normal(f_loc, f_scale).to_event(event_dim),
                                        infer={"is_auxiliary": True})
                    loc = loc + idct(f, dim=-1 - event_dim)
                    stack.enter_context(plates["time"])

                # Sample from a transformed diagonal normal.
                base_dist = dist.Normal(loc, scale).to_event(event_dim)
                pyro.sample(name, dist.TransformedDistribution(base_dist, transform))
