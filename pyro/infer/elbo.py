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
    :param int max_iarange_nesting: optional bound on max number of nested
        :func:`pyro.iarange` contexts. This is only required to enumerate over
        sample sites in parallel, e.g. if a site sets
        ``infer={"enumerate": "parallel"}``.

    References

    [1] `Automated Variational Inference in Probabilistic Programming`
    David Wingate, Theo Weber

    [2] `Black Box Variational Inference`,
    Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def __init__(self,
                 num_particles=1,
                 max_iarange_nesting=float('inf')):
        self.num_particles = num_particles
        self.max_iarange_nesting = max_iarange_nesting

    @staticmethod
    def make(trace_graph=False, enum_discrete=False, **kwargs):
        """
        Factory to construct an ELBO implementation.

        :param bool trace_graph: Whether to keep track of dependency
            information when running the model and guide. This information can
            be used to form a gradient estimator with lower variance in the
            case that some of the random variables are non-reparameterized.
            Note: for a model with many random variables, keeping track of the
            dependency information can be expensive. See the tutorial
            `SVI Part III <http://pyro.ai/examples/svi_part_iii.html>`_ for a
            discussion.
        :param bool enum_discrete: Whether to support summing over discrete
            latent variables, rather than sampling them. To sum out latent
            variables, either wrap the guide in
            :func:`~pyro.infer.enum.config_enumerate` or mark individual sample
            sites with ``infer={"enumerate": "sequential"}`` or
            ``infer={"enumerate": "parallel"}``.
        """
        if trace_graph and enum_discrete:
            raise ValueError("Cannot combine trace_graph with enum_discrete")

        if trace_graph:
            from .tracegraph_elbo import TraceGraph_ELBO
            return TraceGraph_ELBO(**kwargs)
        elif enum_discrete:
            from .traceenum_elbo import TraceEnum_ELBO
            return TraceEnum_ELBO(**kwargs)
        else:
            from .trace_elbo import Trace_ELBO
            return Trace_ELBO(**kwargs)
