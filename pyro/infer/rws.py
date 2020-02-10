# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_if_enumerated, check_model_guide_match, warn_if_nan


class ReweightedWakeSleep(ELBO):
    r"""
    An implementation of Reweighted Wake Sleep following reference [1].

    .. note:: Sampling and log_prob evaluation asymptotic complexity:

    1) Using wake-theta and/or wake-phi
        O(`num_particles`) samples from guide,
        O(`num_particles`) `log_prob` evaluations of model and guide

    2) Using sleep-phi
        O(`num_sleep_particles`) samples from model,
        O(`num_sleep_particles`) `log_prob` evaluations of guide

    if 1) and 2) are combined,
        O(`num_particles`) samples from the guide,
        O(`num_sleep_particles`) from the model,
        O(`num_particles` + `num_sleep_particles`) `log_prob` evaluations of the guide, and
        O(`num_particles`) evaluations of the model

    .. note:: This is particularly useful for models with stochastic branching,
        as described in [2].

    .. note:: This returns _two_ losses, one each for (a) the model parameters (`theta`), computed using the
        `iwae` objective, and (b) the guide parameters (`phi`), computed using (a combination of) the `csis`
        objective and a self-normalized importance-sampled version of the `csis` objective.

    .. note:: In order to enable computing the sleep-phi terms, the guide program must have its observations
        explicitly passed in through the keyworded argument `observations`. Where the value of the observations
        is unknown during definition, such as for amortized variational inference, it may be given a default
        argument as `observations=None`, and the correct value supplied during learning through
        `svi.step(observations=...)`.

    .. warning:: Mini-batch training is not supported yet.

    :param int num_particles: The number of particles/samples used to form the objective
        (gradient) estimator. Default is 2.
    :param insomnia: The scaling between the wake-phi and sleep-phi terms. Default is 1.0 [wake-phi]
    :param bool model_has_params: Indicate if model has learnable params. Useful in avoiding extra
        computation when running in pure sleep mode [csis]. Default is True.
    :param int num_sleep_particles: The number of particles used to form the sleep-phi estimator.
        Matches `num_particles` by default.
    :param bool vectorize_particles: Whether the traces should be vectorised
        across `num_particles`. Default is True.
    :param int max_plate_nesting: Bound on max number of nested
        :func:`pyro.plate` contexts. Default is infinity.
    :param bool strict_enumeration_warning: Whether to warn about possible
        misuse of enumeration, i.e. that
        :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO` is used iff there
        are enumerated sample sites.

    References:

    [1] `Reweighted Wake-Sleep`,
        Jörg Bornschein, Yoshua Bengio

    [2] `Revisiting Reweighted Wake-Sleep for Models with Stochastic Control Flow`,
        Tuan Anh Le, Adam R. Kosiorek, N. Siddharth, Yee Whye Teh, Frank Wood
    """

    def __init__(self,
                 num_particles=2,
                 insomnia=1.,
                 model_has_params=True,
                 num_sleep_particles=None,
                 vectorize_particles=True,
                 max_plate_nesting=float('inf'),
                 strict_enumeration_warning=True):
        # force K > 1 otherwise SNIS not possible
        assert(num_particles > 1), \
            "Reweighted Wake Sleep needs to be run with more than one particle"

        super().__init__(num_particles=num_particles,
                         max_plate_nesting=max_plate_nesting,
                         vectorize_particles=vectorize_particles,
                         strict_enumeration_warning=strict_enumeration_warning)
        self.insomnia = insomnia
        self.model_has_params = model_has_params
        self.num_sleep_particles = num_particles if num_sleep_particles is None else num_sleep_particles

        assert(insomnia >= 0 and insomnia <= 1), \
            "insomnia should be in [0, 1]"

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run against it.
        """
        model_trace, guide_trace = get_importance_trace("flat", self.max_plate_nesting,
                                                        model, guide, args, kwargs, detach=True)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def _loss(self, model, guide, args, kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float, float

        Computes the re-weighted wake-sleep estimators for the model (wake-theta) and the
          guide (insomnia * wake-phi + (1 - insomnia) * sleep-phi).
        Performs backward as appropriate on both, over the specified number of particles.
        """

        wake_theta_loss = torch.tensor(100.)
        if self.model_has_params or self.insomnia > 0.:
            # compute quantities for wake theta and wake phi
            log_joints = []
            log_qs = []

            for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
                log_joint = 0.
                log_q = 0.

                for _, site in model_trace.nodes.items():
                    if site["type"] == "sample":
                        if self.vectorize_particles:
                            log_p_site = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                        else:
                            log_p_site = site["log_prob_sum"]
                        log_joint = log_joint + log_p_site

                for _, site in guide_trace.nodes.items():
                    if site["type"] == "sample":
                        if self.vectorize_particles:
                            log_q_site = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                        else:
                            log_q_site = site["log_prob_sum"]
                        log_q = log_q + log_q_site

                log_joints.append(log_joint)
                log_qs.append(log_q)

            log_joints = log_joints[0] if self.vectorize_particles else torch.stack(log_joints)
            log_qs = log_qs[0] if self.vectorize_particles else torch.stack(log_qs)
            log_weights = log_joints - log_qs.detach()

            # compute wake theta loss
            log_sum_weight = torch.logsumexp(log_weights, dim=0)
            wake_theta_loss = -(log_sum_weight - math.log(self.num_particles)).sum()
            warn_if_nan(wake_theta_loss, "wake theta loss")

        if self.insomnia > 0:
            # compute wake phi loss
            normalised_weights = (log_weights - log_sum_weight).exp().detach()
            wake_phi_loss = -(normalised_weights * log_qs).sum()
            warn_if_nan(wake_phi_loss, "wake phi loss")

        if self.insomnia < 1:
            # compute sleep phi loss
            _model = pyro.poutine.uncondition(model)
            _guide = guide
            _log_q = 0.

            if self.vectorize_particles:
                if self.max_plate_nesting == float('inf'):
                    self._guess_max_plate_nesting(_model, _guide, args, kwargs)
                _model = self._vectorized_num_sleep_particles(_model)
                _guide = self._vectorized_num_sleep_particles(guide)

            for _ in range(1 if self.vectorize_particles else self.num_sleep_particles):
                _model_trace = poutine.trace(_model).get_trace(*args, **kwargs)
                _model_trace.detach_()
                _guide_trace = self._get_matched_trace(_model_trace, _guide, args, kwargs)
                _log_q += _guide_trace.log_prob_sum()

            sleep_phi_loss = -_log_q / self.num_sleep_particles
            warn_if_nan(sleep_phi_loss, "sleep phi loss")

        # compute phi loss
        phi_loss = sleep_phi_loss if self.insomnia == 0 \
            else wake_phi_loss if self.insomnia == 1 \
            else self.insomnia * wake_phi_loss + (1. - self.insomnia) * sleep_phi_loss

        return wake_theta_loss, phi_loss

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float, float

        Computes the re-weighted wake-sleep estimators for the model (wake-theta) and the
          guide (insomnia * wake-phi + (1 - insomnia) * sleep-phi).
        """
        with torch.no_grad():
            wake_theta_loss, phi_loss = self._loss(model, guide, args, kwargs)

        return wake_theta_loss, phi_loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float

        Computes the RWS estimators for the model (wake-theta) and the guide (wake-phi).
        Performs backward as appropriate on both, using num_particle many samples/particles.
        """
        wake_theta_loss, phi_loss = self._loss(model, guide, args, kwargs)
        # convenience addition to ensure easier gradients without requiring `retain_graph=True`
        (wake_theta_loss + phi_loss).backward()

        return wake_theta_loss.detach().item(), phi_loss.detach().item()

    def _vectorized_num_sleep_particles(self, fn):
        """
        Copy of `_vectorised_num_particles` that uses `num_sleep_particles`.
        """
        def wrapped_fn(*args, **kwargs):
            if self.num_sleep_particles == 1:
                return fn(*args, **kwargs)
            with pyro.plate("num_sleep_particles_vectorized", self.num_sleep_particles, dim=-self.max_plate_nesting):
                return fn(*args, **kwargs)

        return wrapped_fn

    @staticmethod
    def _get_matched_trace(model_trace, guide, args, kwargs):
        kwargs["observations"] = {}
        for node in model_trace.stochastic_nodes + model_trace.observation_nodes:
            if "was_observed" in model_trace.nodes[node]["infer"]:
                model_trace.nodes[node]["is_observed"] = True
                kwargs["observations"][node] = model_trace.nodes[node]["value"]

        guide_trace = poutine.trace(poutine.replay(guide, model_trace)).get_trace(*args, **kwargs)
        check_model_guide_match(model_trace, guide_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        return guide_trace
