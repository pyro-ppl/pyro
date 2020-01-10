# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from abc import ABCMeta, abstractmethod

import pyro
import pyro.poutine as poutine
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_site_shape


class ELBO(object, metaclass=ABCMeta):
    """
    :class:`ELBO` is the top-level interface for stochastic variational
    inference via optimization of the evidence lower bound.

    Most users will not interact with this base class :class:`ELBO` directly;
    instead they will create instances of derived classes:
    :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
    :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`, or
    :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO`.

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is only required when enumerating
        over sample sites in parallel, e.g. if a site sets
        ``infer={"enumerate": "parallel"}``. If omitted, ELBO may guess a valid
        value by running the (model,guide) pair once, however this guess may
        be incorrect if model or guide structure is dynamic.
    :param bool vectorize_particles: Whether to vectorize the ELBO computation
        over `num_particles`. Defaults to False. This requires static structure
        in model and guide.
    :param bool strict_enumeration_warning: Whether to warn about possible
        misuse of enumeration, i.e. that
        :class:`pyro.infer.traceenum_elbo.TraceEnum_ELBO` is used iff there
        are enumerated sample sites.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer. When this is True, all :class:`torch.jit.TracerWarning` will
        be ignored. Defaults to False.
    :param bool jit_options: Optional dict of options to pass to
        :func:`torch.jit.trace` , e.g. ``{"check_trace": True}``.
    :param bool retain_graph: Whether to retain autograd graph during an SVI
        step. Defaults to None (False).
    :param float tail_adaptive_beta: Exponent beta with ``-1.0 <= beta < 0.0`` for
        use with `TraceTailAdaptive_ELBO`.

    References

    [1] `Automated Variational Inference in Probabilistic Programming`
    David Wingate, Theo Weber

    [2] `Black Box Variational Inference`,
    Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def __init__(self,
                 num_particles=1,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=False,
                 strict_enumeration_warning=True,
                 ignore_jit_warnings=False,
                 jit_options=None,
                 retain_graph=None,
                 tail_adaptive_beta=-1.0):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting
        self.max_plate_nesting = max_plate_nesting
        self.num_particles = num_particles
        self.vectorize_particles = vectorize_particles
        self.retain_graph = retain_graph
        if self.vectorize_particles and self.num_particles > 1:
            self.max_plate_nesting += 1
        self.strict_enumeration_warning = strict_enumeration_warning
        self.ignore_jit_warnings = ignore_jit_warnings
        self.jit_options = jit_options
        self.tail_adaptive_beta = tail_adaptive_beta

    def _guess_max_plate_nesting(self, model, guide, args, kwargs):
        """
        Guesses max_plate_nesting by running the (model,guide) pair once
        without enumeration. This optimistically assumes static model
        structure.
        """
        # Ignore validation to allow model-enumerated sites absent from the guide.
        with poutine.block():
            guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)
        sites = [site
                 for trace in (model_trace, guide_trace)
                 for site in trace.nodes.values()
                 if site["type"] == "sample"]

        # Validate shapes now, since shape constraints will be weaker once
        # max_plate_nesting is changed from float('inf') to some finite value.
        # Here we know the traces are not enumerated, but later we'll need to
        # allow broadcasting of dims to the left of max_plate_nesting.
        if is_validation_enabled():
            guide_trace.compute_log_prob()
            model_trace.compute_log_prob()
            for site in sites:
                check_site_shape(site, max_plate_nesting=float('inf'))

        dims = [frame.dim
                for site in sites
                for frame in site["cond_indep_stack"]
                if frame.vectorized]
        self.max_plate_nesting = -min(dims) if dims else 0
        if self.vectorize_particles and self.num_particles > 1:
            self.max_plate_nesting += 1
        logging.info('Guessed max_plate_nesting = {}'.format(self.max_plate_nesting))

    def _vectorized_num_particles(self, fn):
        """
        Wraps a callable inside an outermost :class:`~pyro.plate` to parallelize
        ELBO computation over `num_particles`, and to broadcast batch shapes of
        sample site functions in accordance with the `~pyro.plate` contexts
        within which they are embedded.

        :param fn: arbitrary callable containing Pyro primitives.
        :return: wrapped callable.
        """

        def wrapped_fn(*args, **kwargs):
            if self.num_particles == 1:
                return fn(*args, **kwargs)
            with pyro.plate("num_particles_vectorized", self.num_particles, dim=-self.max_plate_nesting):
                return fn(*args, **kwargs)

        return wrapped_fn

    def _get_vectorized_trace(self, model, guide, args, kwargs):
        """
        Wraps the model and guide to vectorize ELBO computation over
        ``num_particles``, and returns a single trace from the wrapped model
        and guide.
        """
        return self._get_trace(self._vectorized_num_particles(model),
                               self._vectorized_num_particles(guide),
                               args, kwargs)

    @abstractmethod
    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        raise NotImplementedError

    def _get_traces(self, model, guide, args, kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.vectorize_particles:
            if self.max_plate_nesting == float('inf'):
                self._guess_max_plate_nesting(model, guide, args, kwargs)
            yield self._get_vectorized_trace(model, guide, args, kwargs)
        else:
            for i in range(self.num_particles):
                yield self._get_trace(model, guide, args, kwargs)
