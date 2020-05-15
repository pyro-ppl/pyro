# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

r"""
The pyro.infer.autoguide.initialization module contains initialization functions for
automatic guides.

The standard interface for initialization is a function that inputs a Pyro
trace ``site`` dict and returns an appropriately sized ``value`` to serve
as an initial constrained value for a guide estimate.
"""
import functools

import torch
from torch.distributions import transform_to

from pyro.distributions.torch import Independent
from pyro.distributions.torch_distribution import MaskedDistribution
from pyro.infer.util import is_validation_enabled
from pyro.poutine.messenger import Messenger
from pyro.util import torch_isnan


# TODO: move this file out of `autoguide` in a minor release

def _is_multivariate(d):
    while isinstance(d, (Independent, MaskedDistribution)):
        d = d.base_dist
    return any(size > 1 for size in d.event_shape)


def init_to_feasible(site=None):
    """
    Initialize to an arbitrary feasible point, ignoring distribution
    parameters.
    """
    if site is None:
        return init_to_feasible

    value = site["fn"].sample().detach()
    t = transform_to(site["fn"].support)
    return t(torch.zeros_like(t.inv(value)))


def init_to_sample(site=None):
    """
    Initialize to a random sample from the prior.
    """
    if site is None:
        return init_to_sample

    return site["fn"].sample().detach()


def init_to_median(site=None, num_samples=15):
    """
    Initialize to the prior median; fallback to a feasible point if median is
    undefined.
    """
    if site is None:
        return functools.partial(init_to_median, num_samples=num_samples)

    # The median undefined for multivariate distributions.
    if _is_multivariate(site["fn"]):
        return init_to_feasible(site)
    try:
        # Try to compute empirical median.
        samples = site["fn"].sample(sample_shape=(num_samples,))
        value = samples.median(dim=0)[0]
        if torch_isnan(value):
            raise ValueError
        if hasattr(site["fn"], "_validate_sample"):
            site["fn"]._validate_sample(value)
        return value
    except (RuntimeError, ValueError):
        # Fall back to feasible point.
        return init_to_feasible(site)


def init_to_mean(site=None):
    """
    Initialize to the prior mean; fallback to median if mean is undefined.
    """
    if site is None:
        return init_to_mean

    try:
        # Try .mean() method.
        value = site["fn"].mean.detach()
        if torch_isnan(value):
            raise ValueError
        if hasattr(site["fn"], "_validate_sample"):
            site["fn"]._validate_sample(value)
        return value
    except (NotImplementedError, ValueError):
        # Fall back to a median.
        # This is required for distributions with infinite variance, e.g. Cauchy.
        return init_to_median(site)


def init_to_uniform(site=None, radius=2):
    """
    Initialize to a random point in the area ``(-radius, radius)`` of
    unconstrained domain.

    :param float radius: specifies the range to draw an initial point in the unconstrained domain.
    """
    if site is None:
        return functools.partial(init_to_uniform, radius=radius)

    value = site["fn"].sample().detach()
    t = transform_to(site["fn"].support)
    return t(torch.rand_like(t.inv(value)) * (2 * radius) - radius)


def init_to_value(site=None, values={}):
    """
    Initialize to the value specified in ``values``. We defer to
    :func:`init_to_uniform` strategy for sites which do not appear in ``values``.

    :param dict values: dictionary of initial values keyed by site name.
    """
    if site is None:
        return functools.partial(init_to_value, values=values)

    if site["name"] in values:
        return values[site["name"]]
    else:
        return init_to_uniform(site)


class _InitToGenerated:
    def __init__(self, generate):
        self.generate = generate
        self._init = None
        self._seen = set()

    def __call__(self, site):
        if self._init is None or site["name"] in self._seen:
            self._init = self.generate()
            self._seen = {site["name"]}
        return self._init(site)


def init_to_generated(site=None, generate=lambda: init_to_uniform):
    """
    Initialize to another initialization strategy returned by the callback
    ``generate`` which is called once per model execution.

    This is like :func:`init_to_value` but can produce different (e.g. random)
    values once per model execution. For example to generate values and return
    ``init_to_value`` you could define::

        def generate():
            values = {"x": torch.randn(100), "y": torch.rand(5)}
            return init_to_value(values=values)

        my_init_fn = init_to_generated(generate=generate)

    :param callable generate: A callable returning another initialization
        function, e.g. returning an ``init_to_value(values={...})`` populated
        with a dictionary of random samples.
    """
    init = _InitToGenerated(generate)
    return init if site is None else init(site)


class InitMessenger(Messenger):
    """
    Initializes a site by replacing ``.sample()`` calls with values
    drawn from an initialization strategy. This is mainly for internal use by
    autoguide classes.

    :param callable init_fn: An initialization function.
    """
    def __init__(self, init_fn):
        self.init_fn = init_fn
        super().__init__()

    def _pyro_sample(self, msg):
        if msg["done"] or msg["is_observed"] or type(msg["fn"]).__name__ == "_Subsample":
            return
        with torch.no_grad():
            value = self.init_fn(msg)
        if is_validation_enabled() and msg["value"] is not None:
            if not isinstance(value, type(msg["value"])):
                raise ValueError(
                    "{} provided invalid type for site {}:\nexpected {}\nactual {}"
                    .format(self.init_fn, msg["name"], type(msg["value"]), type(value)))
            if value.shape != msg["value"].shape:
                raise ValueError(
                    "{} provided invalid shape for site {}:\nexpected {}\nactual {}"
                    .format(self.init_fn, msg["name"], msg["value"].shape, value.shape))
        msg["value"] = value
        msg["done"] = True
