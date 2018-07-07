from __future__ import absolute_import, division, print_function

import math
import warnings

import torch

import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero, log_sum_exp
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, warn_if_nan


class RenyiELBO(ELBO):
    r"""
    An implementation of Renyi's :math:`\alpha`-divergence variational inference
    follows reference [1].

    To have a lower bound, we require :math:`\alpha \ge 0`. However, according to
    reference [1], depending on the dataset, :math:`\alpha < 0` might give better
    results. In the special case :math:`\alpha = 0`, we have important weighted
    lower bound derived in reference [2].

    .. note:: Setting :math:`\alpha < 1` gives a better bound than the usual ELBO.
        For :math:`\alpha = 1`, it is better to use
        :class:`~pyro.infer.trace_elbo.Trace_ELBO` class because it helps reduce
        variances of gradient estimations.

    .. warning:: Mini-batch training is not supported yet.

    :param float alpha: The order of :math:`\alpha`-divergence. Here
        :math:`\alpha \neq 1`. Default is 0.
    :param num_particles: The number of particles/samples used to form the ELBO
        (gradient) estimators. Default is 2.
    :param int max_iarange_nesting: Bound on max number of nested
        :func:`pyro.iarange` contexts. Default is 2.
    :param bool strict_enumeration_warning: Whether to warn about possible
        misuse of enumeration, i.e. that
        :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO` is used iff there
        are enumerated sample sites.

    References:

    [1] `Renyi Divergence Variational Inference`,
        Yingzhen Li, Richard E. Turner

    [2] `Importance Weighted Autoencoders`,
        Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
    """

    def __init__(self,
                 alpha=0,
                 num_particles=1,
                 max_iarange_nesting=float('inf'),
                 vectorize_particles=False,
                 strict_enumeration_warning=True):
        if alpha == 1:
            raise ValueError("The order alpha should not be equal to 1. Please use Trace_ELBO class"
                             "for the case alpha = 1.")
        self.alpha = alpha
        super(RenyiELBO, self).__init__(num_particles, max_iarange_nesting, vectorize_particles,
                                        strict_enumeration_warning)

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
        elbo_list = []
        tensor_holder = None

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0

            # compute elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    if self.vectorize_particles and self.num_particles > 1:
                        log_prob_sum = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]
                    elbo_particle = elbo_particle + log_prob_sum.detach()

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    if self.vectorize_particles and self.num_particles > 1:
                        log_prob_sum = log_prob.reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]

                    elbo_particle = elbo_particle - log_prob_sum.detach()

            if is_identically_zero(elbo_particle):
                if tensor_holder is not None:
                    elbo_particle = tensor_holder.new_zeros(tensor_holder.shape)
            else:  # elbo_particle is not None
                if tensor_holder is None:
                    tensor_holder = elbo_particle.new_empty(elbo_particle.shape)
                    # change types of previous `elbo_particle`s
                    for i in range(len(elbo_list)):
                        elbo_list[i] = tensor_holder.new_zeros(tensor_holder.shape)

            elbo_list.append(elbo_particle)

        if tensor_holder is None:
            return 0.

        if (not self.vectorize_particles) and self.num_particles > 1:
            elbo_particles = torch.stack(elbo_list)
        else:
            elbo_particles = elbo_list[0]

        elbo_particles_scaled = (1. - self.alpha) * elbo_particles
        if self.num_particles == 1:
            elbo_scaled = elbo_particles_scaled
        else:
            elbo_scaled = log_sum_exp(elbo_particle_scaled, dim=0) - math.log(self.num_particles)

        loss = elbo_scaled.sum().item() / (self.alpha - 1.)
        warn_if_nan(loss, "loss")
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        elbo_list = []
        surrogate_elbo_list = []
        tensor_holder = None

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0

            # compute elbo and surrogate elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    if self.vectorize_particles and self.num_particles > 1:
                        log_prob_sum = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]
                    elbo_particle = elbo_particle + log_prob_sum.detach()
                    surrogate_elbo_particle = surrogate_elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    if self.vectorize_particles and self.num_particles > 1:
                        log_prob_sum = log_prob.reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]

                    elbo_particle = elbo_particle - log_prob_sum.detach()

                    if not is_identically_zero(entropy_term):
                        surrogate_elbo_particle = surrogate_elbo_particle - log_prob_sum

                    if not is_identically_zero(score_function_term):
                        surrogate_elbo_particle = (surrogate_elbo_particle +
                                                   (self.alpha / (1. - self.alpha)) * log_prob_sum)

            if is_identically_zero(elbo_particle):
                if tensor_holder is not None:
                    elbo_particle = tensor_holder.new_zeros(tensor_holder.shape)
                    surrogate_elbo_particle = tensor_holder.new_zeros(tensor_holder.shape)
            else:  # elbo_particle is not None
                if tensor_holder is None:
                    tensor_holder = elbo_particle.new_empty(elbo_particle.shape)
                    # change types of previous `elbo_particle`s
                    for i in range(len(elbo_list)):
                        elbo_list[i] = tensor_holder.new_zeros(tensor_holder.shape)
                        surrogate_elbo_list[i] = tensor_holder.new_zeros(tensor_holder.shape)

            elbo_list.append(elbo_particle)
            surrogate_elbo_list.append(surrogate_elbo_particle)

        if tensor_holder is None:
            return 0.

        if (not self.vectorize_particles) and self.num_particles > 1:
            elbo_particles = torch.stack(elbo_list)
            surrogate_elbo_particles = torch.stack(surrogate_elbo_list)
        else:
            elbo_particles = elbo_list[0]
            surrogate_elbo_particles = surrogate_elbo_list[0]

        elbo_particles_scaled = (1. - self.alpha) * elbo_particles
        if self.num_particles == 1:
            elbo_scaled = elbo_particles_scaled
        else:
            elbo_scaled = log_sum_exp(elbo_particles_scaled, dim=0) - math.log(self.num_particles)

        # collect parameters to train from model and guide
        trainable_params = any(site["type"] == "param"
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values())

        if trainable_params and getattr(surrogate_elbo_particles, 'requires_grad', False):
            if self.num_particles == 1:
                weights = 1.
            else:
                weights = (elbo_particles_scaled - elbo_scaled).exp()

            surrogate_loss = - (weights * surrogate_elbo_particles).sum() / self.num_particles
            surrogate_loss.backward()

        loss = elbo_scaled.sum().item() / (self.alpha - 1.)
        warn_if_nan(loss, "loss")
        return loss
