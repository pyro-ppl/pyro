# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import torch

from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import get_dependent_plate_dims, is_validation_enabled, torch_sum
from pyro.util import check_if_enumerated, warn_if_nan


class RenyiELBO(ELBO):
    r"""
    An implementation of Renyi's :math:`\alpha`-divergence variational inference
    following reference [1].

    In order for the objective to be a strict lower bound, we require
    :math:`\alpha \ge 0`. Note, however, that according to reference [1], depending
    on the dataset :math:`\alpha < 0` might give better results. In the special case
    :math:`\alpha = 0`, the objective function is that of the important weighted
    autoencoder derived in reference [2].

    .. note:: Setting :math:`\alpha < 1` gives a better bound than the usual ELBO.
        For :math:`\alpha = 1`, it is better to use
        :class:`~pyro.infer.trace_elbo.Trace_ELBO` class because it helps reduce
        variances of gradient estimations.

    :param float alpha: The order of :math:`\alpha`-divergence. Here
        :math:`\alpha \neq 1`. Default is 0.
    :param num_particles: The number of particles/samples used to form the objective
        (gradient) estimator. Default is 2.
    :param int max_plate_nesting: Bound on max number of nested
        :func:`pyro.plate` contexts. Default is infinity.
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
                 num_particles=2,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=False,
                 strict_enumeration_warning=True):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting

        if alpha == 1:
            raise ValueError("The order alpha should not be equal to 1. Please use Trace_ELBO class"
                             "for the case alpha = 1.")
        self.alpha = alpha
        super().__init__(num_particles=num_particles,
                         max_plate_nesting=max_plate_nesting,
                         vectorize_particles=vectorize_particles,
                         strict_enumeration_warning=strict_enumeration_warning)

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo_particles = []
        is_vectorized = self.vectorize_particles and self.num_particles > 1

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = 0.
            sum_dims = get_dependent_plate_dims(model_trace.nodes.values())

            # compute elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob_sum = torch_sum(site["log_prob"], sum_dims)
                    elbo_particle = elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    log_prob_sum = torch_sum(site["log_prob"], sum_dims)
                    elbo_particle = elbo_particle - log_prob_sum

            elbo_particles.append(elbo_particle)

        if is_vectorized:
            elbo_particles = elbo_particles[0]
        else:
            elbo_particles = torch.stack(elbo_particles)

        log_weights = (1. - self.alpha) * elbo_particles
        log_mean_weight = torch.logsumexp(log_weights, dim=0) - math.log(self.num_particles)
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
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0
            sum_dims = get_dependent_plate_dims(model_trace.nodes.values())

            # compute elbo and surrogate elbo
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob_sum = torch_sum(site["log_prob"], sum_dims)
                    elbo_particle = elbo_particle + log_prob_sum.detach()
                    surrogate_elbo_particle = surrogate_elbo_particle + log_prob_sum

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_prob, score_function_term, entropy_term = site["score_parts"]
                    log_prob_sum = torch_sum(site["log_prob"], sum_dims)

                    elbo_particle = elbo_particle - log_prob_sum.detach()

                    if not is_identically_zero(entropy_term):
                        surrogate_elbo_particle = surrogate_elbo_particle - log_prob_sum

                        if not is_identically_zero(score_function_term):
                            # link to the issue: https://github.com/pyro-ppl/pyro/issues/1222
                            raise NotImplementedError

                    if not is_identically_zero(score_function_term):
                        surrogate_elbo_particle = (surrogate_elbo_particle +
                                                   (self.alpha / (1. - self.alpha)) * log_prob_sum)

            if is_identically_zero(elbo_particle):
                if tensor_holder is not None:
                    elbo_particle = torch.zeros_like(tensor_holder)
                    surrogate_elbo_particle = torch.zeros_like(tensor_holder)
            else:  # elbo_particle is not None
                if tensor_holder is None:
                    tensor_holder = torch.zeros_like(elbo_particle)
                    # change types of previous `elbo_particle`s
                    for i in range(len(elbo_particles)):
                        elbo_particles[i] = torch.zeros_like(tensor_holder)
                        surrogate_elbo_particles[i] = torch.zeros_like(tensor_holder)

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
        log_mean_weight = torch.logsumexp(log_weights, dim=0, keepdim=True) - math.log(self.num_particles)
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
