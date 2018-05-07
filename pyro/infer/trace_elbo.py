from __future__ import absolute_import, division, print_function

import warnings
import weakref

import pyro
import pyro.ops.jit
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.util import MultiFrameTensor, get_iarange_stacks, is_validation_enabled, torch_item
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, torch_isnan


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_iarange_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r


class Trace_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.iarange` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """
        for i in range(self.num_particles):
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

            yield model_trace, guide_trace

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
        elbo = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0
            log_r = None

            # compute elbo and surrogate elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                    surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]

                    elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                    if not is_identically_zero(entropy_term):
                        surrogate_elbo_particle = surrogate_elbo_particle - entropy_term.sum()

                    if not is_identically_zero(score_function_term):
                        if log_r is None:
                            log_r = _compute_log_r(model_trace, guide_trace)
                        site = log_r.sum_to(site["cond_indep_stack"])
                        surrogate_elbo_particle = surrogate_elbo_particle + (site * score_function_term).sum()

            elbo += elbo_particle / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(site["type"] == "param"
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values())

            if trainable_params and getattr(surrogate_elbo_particle, 'requires_grad', False):
                surrogate_loss_particle = -surrogate_elbo_particle / self.num_particles
                surrogate_loss_particle.backward()

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss


class JitTrace_ELBO(Trace_ELBO):
    """
    Like :class:`Trace_ELBO` but uses :func:`pyro.ops.jit.compile` to compile
    :meth:`loss_and_grads`.

    This works only for a limited set of models:

    -   Models must have static structure.
    -   Models must not depend on any global data (except the param store).
    -   All model inputs that are tensors must be passed in via ``*args``.
    -   All model inputs that are *not* tensors must be passed in via
        ``*kwargs``, and these will be fixed to their values on the first
        call to :meth:`jit_loss_and_grads`.

    .. warning:: Experimental. Interface subject to change.
    """
    def loss_and_grads(self, model, guide, *args, **kwargs):
        if getattr(self, '_loss_and_surrogate_loss', None) is None:
            # build a closure for loss_and_surrogate_loss
            weakself = weakref.ref(self)

            @pyro.ops.jit.compile(nderivs=1)
            def loss_and_surrogate_loss(*args):
                self = weakself()
                loss = 0.0
                surrogate_loss = 0.0
                for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
                    elbo_particle = 0
                    surrogate_elbo_particle = 0
                    log_r = None

                    # compute elbo and surrogate elbo
                    for name, site in model_trace.nodes.items():
                        if site["type"] == "sample":
                            elbo_particle = elbo_particle + site["log_prob_sum"]
                            surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]

                    for name, site in guide_trace.nodes.items():
                        if site["type"] == "sample":
                            log_prob, score_function_term, entropy_term = site["score_parts"]

                            elbo_particle = elbo_particle - site["log_prob_sum"]

                            if not is_identically_zero(entropy_term):
                                surrogate_elbo_particle = surrogate_elbo_particle - entropy_term.sum()

                            if not is_identically_zero(score_function_term):
                                if log_r is None:
                                    log_r = _compute_log_r(model_trace, guide_trace)
                                site = log_r.sum_to(site["cond_indep_stack"])
                                surrogate_elbo_particle = surrogate_elbo_particle + (site * score_function_term).sum()

                    loss = loss - elbo_particle / self.num_particles
                    surrogate_loss = surrogate_loss - surrogate_elbo_particle / self.num_particles

                return loss, surrogate_loss

            self._loss_and_surrogate_loss = loss_and_surrogate_loss

        # invoke _loss_and_surrogate_loss
        loss, surrogate_loss = self._loss_and_surrogate_loss(*args)
        surrogate_loss.backward()  # this line triggers jit compilation
        loss = loss.item()

        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
