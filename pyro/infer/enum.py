# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numbers
from functools import partial
from queue import LifoQueue

from pyro import poutine
from pyro.infer.util import is_validation_enabled
from pyro.poutine import Trace
from pyro.poutine.enum_messenger import enumerate_site
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, ignore_jit_warnings


def iter_discrete_escape(trace, msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            (msg["infer"].get("enumerate") == "sequential") and  # only sequential
            (msg["name"] not in trace))


def iter_discrete_extend(trace, site, **ignored):
    values = enumerate_site(site)
    enum_total = values.shape[0]
    with ignore_jit_warnings(["Converting a tensor to a Python index",
                              ("Iterating over a tensor", RuntimeWarning)]):
        values = iter(values)
    for i, value in enumerate(values):
        extended_site = site.copy()
        extended_site["infer"] = site["infer"].copy()
        extended_site["infer"]["_enum_total"] = enum_total
        extended_site["value"] = value
        extended_trace = trace.copy()
        extended_trace.add_node(site["name"], **extended_site)
        yield extended_trace


def get_importance_trace(graph_type, max_plate_nesting, model, guide, args, kwargs, detach=False):
    """
    Returns a single trace from the guide, which can optionally be detached,
    and the model that is run against it.
    """
    guide_trace = poutine.trace(guide, graph_type=graph_type).get_trace(*args, **kwargs)
    if detach:
        guide_trace.detach_()
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace),
                                graph_type=graph_type).get_trace(*args, **kwargs)
    if is_validation_enabled():
        check_model_guide_match(model_trace, guide_trace, max_plate_nesting)

    guide_trace = prune_subsample_sites(guide_trace)
    model_trace = prune_subsample_sites(model_trace)

    model_trace.compute_log_prob()
    guide_trace.compute_score_parts()
    if is_validation_enabled():
        for site in model_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, max_plate_nesting)
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, max_plate_nesting)

    return model_trace, guide_trace


def iter_discrete_traces(graph_type, fn, *args, **kwargs):
    """
    Iterate over all discrete choices of a stochastic function.

    When sampling continuous random variables, this behaves like `fn`.
    When sampling discrete random variables, this iterates over all choices.

    This yields traces scaled by the probability of the discrete choices made
    in the `trace`.

    :param str graph_type: The type of the graph, e.g. "flat" or "dense".
    :param callable fn: A stochastic function.
    :returns: An iterator over traces pairs.
    """
    queue = LifoQueue()
    queue.put(Trace())
    traced_fn = poutine.trace(
        poutine.queue(fn, queue, escape_fn=iter_discrete_escape, extend_fn=iter_discrete_extend),
        graph_type=graph_type)
    while not queue.empty():
        yield traced_fn.get_trace(*args, **kwargs)


def _config_fn(default, expand, num_samples, tmc, site):
    if site["type"] != "sample" or site["is_observed"]:
        return {}
    if type(site["fn"]).__name__ == "_Subsample":
        return {}
    if num_samples is not None:
        return {"enumerate": site["infer"].get("enumerate", default),
                "num_samples": site["infer"].get("num_samples", num_samples),
                "expand": site["infer"].get("expand", expand),
                "tmc": site["infer"].get("tmc", tmc)}
    if getattr(site["fn"], "has_enumerate_support", False):
        return {"enumerate": site["infer"].get("enumerate", default),
                "expand": site["infer"].get("expand", expand)}
    return {}


def _config_enumerate(default, expand, num_samples, tmc):
    return partial(_config_fn, default, expand, num_samples, tmc)


def config_enumerate(guide=None, default="parallel", expand=False, num_samples=None, tmc="diagonal"):
    """
    Configures enumeration for all relevant sites in a guide. This is mainly
    used in conjunction with :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO`.

    When configuring for exhaustive enumeration of discrete variables, this
    configures all sample sites whose distribution satisfies
    ``.has_enumerate_support == True``.
    When configuring for local parallel Monte Carlo sampling via
    ``default="parallel", num_samples=n``, this configures all sample sites.
    This does not overwrite existing annotations ``infer={"enumerate": ...}``.

    This can be used as either a function::

        guide = config_enumerate(guide)

    or as a decorator::

        @config_enumerate
        def guide1(*args, **kwargs):
            ...

        @config_enumerate(default="sequential", expand=True)
        def guide2(*args, **kwargs):
            ...

    :param callable guide: a pyro model that will be used as a guide in
        :class:`~pyro.infer.svi.SVI`.
    :param str default: Which enumerate strategy to use, one of
        "sequential", "parallel", or None. Defaults to "parallel".
    :param bool expand: Whether to expand enumerated sample values. See
        :meth:`~pyro.distributions.Distribution.enumerate_support` for details.
        This only applies to exhaustive enumeration, where ``num_samples=None``.
        If ``num_samples`` is not ``None``, then this samples will always be
        expanded.
    :param num_samples: if not ``None``, use local Monte Carlo sampling rather
        than exhaustive enumeration. This makes sense for both continuous and
        discrete distributions.
    :type num_samples: int or None
    :param tmc: "mixture" or "diagonal" strategies to use in Tensor Monte Carlo
    :type tmc: string or None
    :return: an annotated guide
    :rtype: callable
    """
    if default not in ["sequential", "parallel", "flat", None]:
        raise ValueError("Invalid default value. Expected 'sequential', 'parallel', or None, but got {}".format(
            repr(default)))
    if expand not in [True, False]:
        raise ValueError("Invalid expand value. Expected True or False, but got {}".format(repr(expand)))
    if num_samples is not None:
        if not (isinstance(num_samples, numbers.Number) and num_samples > 0):
            raise ValueError("Invalid num_samples, expected None or positive integer, but got {}".format(
                repr(num_samples)))
        if default == "sequential":
            raise ValueError('Local sampling does not support "sequential" sampling; '
                             'use "parallel" sampling instead.')
    if tmc == "full" and num_samples is not None and num_samples > 1:
        # tmc strategies validated elsewhere (within enum handler)
        expand = True

    # Support usage as a decorator:
    if guide is None:
        return lambda guide: config_enumerate(guide, default=default, expand=expand, num_samples=num_samples, tmc=tmc)

    return poutine.infer_config(guide, config_fn=_config_enumerate(default, expand, num_samples, tmc))
