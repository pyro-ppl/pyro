from __future__ import absolute_import, division, print_function

import warnings
from abc import ABCMeta, abstractmethod

from six import add_metaclass

import pyro


@add_metaclass(ABCMeta)
class ELBO(object):
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
        :func:`pyro.plate` contexts. This is only required to enumerate over
        sample sites in parallel, e.g. if a site sets
        ``infer={"enumerate": "parallel"}``.
    :param bool vectorize_particles: Whether to vectorize the ELBO computation
        over `num_particles`. Defaults to False. This requires static structure
        in model and guide. In addition, this requires specifying a finite
        value for `max_plate_nesting`.
    :param bool strict_enumeration_warning: Whether to warn about possible
        misuse of enumeration, i.e. that
        :class:`pyro.infer.traceenum_elbo.TraceEnum_ELBO` is used iff there
        are enumerated sample sites.
    :param bool retain_graph: Whether to retain autograd graph during an SVI step.
        Defaults to None (False).

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
                 retain_graph=None):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting

        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting
        self.vectorize_particles = vectorize_particles
        self.retain_graph = retain_graph
        if self.vectorize_particles:
            if self.num_particles > 1:
                if self.max_plate_nesting == float('inf'):
                    raise ValueError("Automatic vectorization over num_particles requires " +
                                     "a finite value for `max_plate_nesting` arg.")
                self.max_plate_nesting += 1
        self.strict_enumeration_warning = strict_enumeration_warning

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

    def _get_vectorized_trace(self, model, guide, *args, **kwargs):
        """
        Wraps the model and guide to vectorize ELBO computation over
        ``num_particles``, and returns a single trace from the wrapped model
        and guide.
        """
        return self._get_trace(self._vectorized_num_particles(model),
                               self._vectorized_num_particles(guide),
                               *args, **kwargs)

    @abstractmethod
    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        raise NotImplementedError

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.vectorize_particles:
            yield self._get_vectorized_trace(model, guide, *args, **kwargs)
        else:
            for i in range(self.num_particles):
                yield self._get_trace(model, guide, *args, **kwargs)
