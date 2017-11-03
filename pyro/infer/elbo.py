from __future__ import absolute_import, division, print_function

from .trace_elbo import Trace_ELBO
from .tracegraph_elbo import TraceGraph_ELBO


class ELBO(object):
    """
    :param num_particles: the number of particles (samples) used to form the ELBO estimator.
    :param trace_graph: boolean. whether to keep track of dependency information when running the
        model and guide. this information can be used to form a gradient estimator with lower variance
        in the case that some of the random variables are non-reparameterized.
        note: for a model with many random variables, keeping track of the dependency information
        can be expensive. see the tutorial `SVI Part III <http://pyro.ai/examples/svi_part_iii.html>`_
        for a discussion.
    :param bool enum_discrete: whether to sum over discrete latent variables, rather than sample them.

    `ELBO` is the top-level interface for stochastic variational inference via optimization of the
    evidence lower bound. Most users will not interact with `ELBO` directly; instead they will interact
    with `SVI`. `ELBO` dispatches to `Trace_ELBO` and `TraceGraph_ELBO`, where the internal
    implementations live.

    .. warning:: `enum_discrete` is a bleeding edge feature.
        see `SS-VAE <http://pyro.ai/examples/ss_vae.html>`_ for a discussion.

    References

    [1] `Automated Variational Inference in Probabilistic Programming`
    David Wingate, Theo Weber

    [2] `Black Box Variational Inference`,
    Rajesh Ranganath, Sean Gerrish, David M. Blei
    """
    def __init__(self,
                 num_particles=1,
                 trace_graph=False,
                 enum_discrete=False):
        super(ELBO, self).__init__()
        self.num_particles = num_particles
        self.trace_graph = trace_graph
        if self.trace_graph:
            self.which_elbo = TraceGraph_ELBO(num_particles=num_particles, enum_discrete=enum_discrete)
        else:
            self.which_elbo = Trace_ELBO(num_particles=num_particles, enum_discrete=enum_discrete)

    def loss(self, model, guide, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses `num_particles` many samples/particles,
        where `num_particles` is specified in the constructor.

        :returns: returns an estimate of the ELBO
        :rtype: float
        """
        return self.which_elbo.loss(model, guide, *args, **kwargs)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators,
        where `num_particles` is specified in the constructor.

        :returns: returns an estimate of the ELBO
        :rtype: float
        """
        return self.which_elbo.loss_and_grads(model, guide, *args, **kwargs)
