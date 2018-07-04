from __future__ import absolute_import, division, print_function

import math
import warnings

import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero, log_sum_exp
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, torch_isnan


class RenyiELBO(ELBO):
    u"""
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide.

    References:

    [1] `R\00e9nyi Divergence Variational Inference`,
        Yingzhen Li, Richard E. Turner

    [2] `Importance Weighted Autoencoders`,
        Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
    """
    
    def __init__(self, num_particles=1, max_iarange_nesting=float('inf'),
                 strict_enumeration_warning=True):
        super(RenyiELBO, self).__init__(num_particles, max_iarange_nesting, vectorize_particle=True,
                                        strict_enumeration_warning=True)

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        if is_validation_enabled():
            check_model_guide_match(model_trace, guide_trace)
            enumerated_sites = [name for name, site in guide_trace.nodes.items()
                                if site["type"] == "sample" and site["infer"].get("enumerate")]
            if enumerated_sites:
                warnings.warn('\n'.join([
                    'Trace_ELBO found sample sites configured for enumeration:'
                    ', '.join(enumerated_sites),
                    'If you want to enumerate sites, you need to use TraceEnum_ELBO instead.']))
        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)

        model_trace.compute_log_prob()
        guide_trace.compute_score_parts()
        if is_validation_enabled():
            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)

        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(guide_trace.log_prob_sum())
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        # grab a vectorized trace from the generator
        model_trace, guide_trace = self._get_traces(model, guide, *args, **kwargs).next()
        elbo_particle = 0
        surrogate_elbo_particle = 0
        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle = elbo_particle + site["log_prob"].detach()
                surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob"]

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - log_prob.detach()

                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = surrogate_elbo_particle - entropy_term

                if not is_identically_zero(score_function_term):
                    surrogate_elbo_particle = surrogate_elbo_particle + (self.alpha / (1. - self.alpha)) * score_function_term

        if is_identically_zero(elbo_particle):
            return 0.

        elbo_particle_scaled = (1. - self.alpha) * elbo_particle
        if self.num_particles == 1:
            elbo_scaled = elbo_particle_scaled
        else:
            elbo_scaled = log_sum_exp(elbo_particle_scaled, dim=0) - math.log(self.num_particles)

        # collect parameters to train from model and guide
        trainable_params = any(site["type"] == "param"
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values())

        if trainable_params and getattr(surrogate_elbo_particle, 'requires_grad', False):
            if self.num_particles == 1:
                weights = 1.
            else:
                weights = torch.exp(elbo_particle_scaled - elbo_scaled)
            surrogate_loss = - (weights * surrogate_elbo_particle).sum() / self.num_particles
            surrogate_loss.backward()

        loss = elbo_scaled.sum().item() / (self.alpha - 1.)
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
