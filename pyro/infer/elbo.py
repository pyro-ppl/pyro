from __future__ import absolute_import, division, print_function


class ELBO(object):
    """
    :class:`ELBO` is the top-level interface for stochastic variational
    inference via optimization of the evidence lower bound. Most users will not
    interact with :class:`ELBO` directly; instead they will interact with `SVI`.
    `ELBO` dispatches to `Trace_ELBO` and `TraceGraph_ELBO`, where the internal
    implementations live.

    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators.
    :param int max_iarange_nesting: Optional bound on max number of nested
        :func:`pyro.iarange` contexts. This is only required to enumerate over
        sample sites in parallel, e.g. if a site sets
        ``infer={"enumerate": "parallel"}``.
    :param bool strict_enumeration_warning: Whether to warn about possible
        misuse of enumeration, i.e. that
        :class:`pyro.infer.traceenum_elbo.TraceEnum_ELBO` is used iff there
        are enumerated sample sites.

    References

    [1] `Automated Variational Inference in Probabilistic Programming`
    David Wingate, Theo Weber

    [2] `Black Box Variational Inference`,
    Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def __init__(self,
                 num_particles=1,
                 max_iarange_nesting=float('inf'),
                 strict_enumeration_warning=True):
        self.num_particles = num_particles
        self.max_iarange_nesting = max_iarange_nesting
        self.strict_enumeration_warning = strict_enumeration_warning
