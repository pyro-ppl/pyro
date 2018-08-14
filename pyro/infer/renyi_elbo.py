from __future__ import absolute_import, division, print_function

import math

import torch

from pyro.distributions.util import is_identically_zero, logsumexp
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan


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
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_iarange_nesting, model, guide, *args, **kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0.

            # compute elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    if is_vectorized:
                        log_prob_sum = site["log_prob"].detach().reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = torch_item(site["log_prob_sum"])

                    elbo_particle = elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    if is_vectorized:
                        log_prob_sum = log_prob.detach().reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = torch_item(site["log_prob_sum"])

                    elbo_particle = elbo_particle - log_prob_sum

            elbo_particles.append(elbo_particle)

        if is_vectorized:
            elbo_particles = elbo_particles[0]
        else:
            elbo_particles = torch.tensor(elbo_particles)  # no need to use .new*() here

        log_weights = (1. - self.alpha) * elbo_particles
        log_mean_weight = logsumexp(log_weights, dim=0) - math.log(self.num_particles)
        elbo = log_mean_weight.sum().item() / (1. - self.alpha)

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        elbo_particles = []
        surrogate_elbo_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1
        tensor_holder = None

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0

            # compute elbo and surrogate elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    if is_vectorized:
                        log_prob_sum = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]
                    elbo_particle = elbo_particle + log_prob_sum.detach()
                    surrogate_elbo_particle = surrogate_elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    if is_vectorized:
                        log_prob_sum = log_prob.reshape(self.num_particles, -1).sum(-1)
                    else:
                        log_prob_sum = site["log_prob_sum"]

                    elbo_particle = elbo_particle - log_prob_sum.detach()

                    if not is_identically_zero(entropy_term):
                        surrogate_elbo_particle = surrogate_elbo_particle - log_prob_sum

                        if not is_identically_zero(score_function_term):
                            # link to the issue: https://github.com/uber/pyro/issues/1222
                            raise NotImplementedError

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
                    for i in range(len(elbo_particles)):
                        elbo_particles[i] = tensor_holder.new_zeros(tensor_holder.shape)
                        surrogate_elbo_particles[i] = tensor_holder.new_zeros(tensor_holder.shape)

            elbo_particles.append(elbo_particle)
            surrogate_elbo_particles.append(surrogate_elbo_particle)

        if tensor_holder is None:
            return 0.

        if is_vectorized:
            elbo_particles = elbo_particles[0]
            surrogate_elbo_particles = surrogate_elbo_particles[0]
        else:
            elbo_particles = torch.stack(elbo_particles)
            surrogate_elbo_particles = torch.stack(surrogate_elbo_particles)

        log_weights = (1. - self.alpha) * elbo_particles
        log_mean_weight = logsumexp(log_weights, dim=0) - math.log(self.num_particles)
        elbo = log_mean_weight.sum().item() / (1. - self.alpha)

        # collect parameters to train from model and guide
        trainable_params = any(site["type"] == "param"
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values())

        if trainable_params and getattr(surrogate_elbo_particles, 'requires_grad', False):
            normalized_weights = (log_weights - log_mean_weight).exp()
            surrogate_elbo = (normalized_weights * surrogate_elbo_particles).sum() / self.num_particles
            surrogate_loss = -surrogate_elbo
            surrogate_loss.backward()

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss
