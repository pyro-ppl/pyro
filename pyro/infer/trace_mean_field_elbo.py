from __future__ import absolute_import, division, print_function

from torch.distributions import kl_divergence

from pyro.distributions.util import is_identically_zero
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.util import torch_item
from pyro.util import warn_if_nan


class TraceMeanField_ELBO(Trace_ELBO):
    """
    A trace implementation of ELBO-based SVI. This is currently the only
    ELBO estimator in Pyro that uses analytic KL divergences when those
    are available.

    In contrast to, e.g.,
    :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO` and
    :class:`~pyro.infer.tracegraph_elbo.Trace_ELBO` this estimator places
    restrictions on the dependency structure of the model and guide.
    In particular it assumes that the guide has a mean-field structure,
    i.e. that it factorizes across the different latent variables present
    in the guide. It also assumes that all of the latent variables in the
    guide are reparameterized. This latter condition is satisfied for, e.g.,
    the Normal distribution but is not satisfied for, e.g., the Categorical
    distribution.

    .. warning:: This estimator may give incorrect results if the mean-field
      condition is not satisfied.

    Note for advanced users:

    The mean field condition is a sufficient but not necessary condition for
    this estimator to be correct. The precise condition is that for every
    latent variable `z` in the guide, its parents in the model must not include
    any latent variables that are descendants of `z` in the guide. Here
    'parents in the model' and 'descendants in the guide' is with respect
    to the corresponding (statistical) dependency structure. For example, this
    condition is always satisfied if the model and guide have identical
    dependency structures.
    """

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle, _ = self._differentiable_loss_particle(model_trace, guide_trace)
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0

        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                else:
                    guide_site = guide_trace.nodes[name]

                    log_prob, score_function_term, entropy_term = guide_site["score_parts"]
                    fully_rep = guide_site["fn"].has_rsample and not is_identically_zero(entropy_term) \
                        and is_identically_zero(score_function_term)
                    assert fully_rep, "All distributions in the guide must be fully reparameterized."

                    # use kl divergence if available, else fall back on sampling
                    try:
                        kl_qp = kl_divergence(guide_site["fn"], model_site["fn"])
                        assert kl_qp.shape == guide_site["fn"].batch_shape
                        elbo_particle = elbo_particle - kl_qp.sum()
                    except NotImplementedError:
                        elbo_particle = elbo_particle + model_site["log_prob_sum"] - entropy_term.sum()

        # handle auxiliary sites in the guide
        for name, guide_site in guide_trace.nodes.items():
            if guide_site["type"] == "sample" and name not in model_trace.nodes():
                assert guide_site["infer"].get("is_auxiliary")

                log_prob, score_function_term, entropy_term = guide_site["score_parts"]
                fully_rep = guide_site["fn"].has_rsample and not is_identically_zero(entropy_term) \
                    and is_identically_zero(score_function_term)
                assert fully_rep, "All distributions in the guide must be fully reparameterized."

                elbo_particle = elbo_particle - entropy_term.sum()

        return -torch_item(elbo_particle), -elbo_particle
