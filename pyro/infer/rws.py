import math
import warnings

import torch

from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace_detached
from pyro.infer.util import is_validation_enabled, torch_item
from pyro.util import check_if_enumerated, warn_if_nan


class ReweightedWakeSleep(ELBO):
    r"""
    An implementation of Reweighted Wake Sleep following reference [1].

    .. note:: This is particularly useful for models with stochastic branching,
        as described in [2].

    .. note:: This returns _two_ losses, one each for the model and the guide.

    .. warning:: Mini-batch training is not supported yet.

    :param num_particles: The number of particles/samples used to form the objective
        (gradient) estimator. Default is 2.
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
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=True,
                 strict_enumeration_warning=True):
        if max_iarange_nesting is not None:
            warnings.warn("max_iarange_nesting is deprecated; use max_plate_nesting instead",
                          DeprecationWarning)
            max_plate_nesting = max_iarange_nesting

        # force K > 1 otherwise SNIS not possible
        assert(num_particles > 1), \
            "Reweighted Wake Sleep needs to be run with more than one particle"

        super(ReweightedWakeSleep, self).__init__(num_particles=num_particles,
                                                  max_plate_nesting=max_plate_nesting,
                                                  vectorize_particles=vectorize_particles,
                                                  strict_enumeration_warning=strict_enumeration_warning)

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run against it.
        """

        # RWS: this samples detached zs
        model_trace, guide_trace = get_importance_trace_detached(
            "flat", self.max_plate_nesting, model, guide, *args, **kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates RWS with estimators that uses num_particles many samples/particles.
        """
        log_joints = []
        log_qs = []

        # grab a vectorized trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            log_joint = 0.
            log_q = 0.

            # compute log weights
            for _, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    log_p_site = site["log_prob"].detach()
                    log_joint = log_joint + log_p_site

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_q_site = site["log_prob"].detach()
                    log_q = log_q + log_q_site

            log_joints.append(log_joint)
            log_qs.append(log_q)

        log_joints = log_joints[0] if self.vectorize_particles else torch.stack(log_joints)
        log_qs = log_qs[0] if self.vectorize_particles else torch.stack(log_qs)
        log_weights = log_joints - log_qs

        # wake theta = iwae:
        log_sum_weight = torch.logsumexp(log_weights, dim=0)
        wake_theta_loss = -(log_sum_weight - math.log(self.num_particles)).sum()

        # wake phi = reweighted csis:
        normalised_weights = (log_weights - log_sum_weight).exp()
        wake_phi_loss = -(normalised_weights * log_qs).sum(0).sum(0)

        warn_if_nan(wake_theta_loss, "loss")
        warn_if_nan(wake_phi_loss, "loss")
        return wake_theta_loss, wake_phi_loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns model loss and guide loss
        :rtype: float

        Computes the RWS estimators for the model (wake-theta) and the guide (wake-phi).
        Performs backward as appropriate on both, using num_particle many samples/particles.
        """
        log_joints = []
        log_qs = []

        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            log_joint = 0.
            log_q = 0.

            # compute log_weight and log_q
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    log_p_site = site["log_prob"]
                    log_joint = log_joint + log_p_site

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    log_q_site = site["log_prob"]
                    log_q = log_q + log_q_site

            log_joints.append(log_joint)
            log_qs.append(log_q)

        log_joints = log_joints[0] if self.vectorize_particles else torch.stack(log_joints)
        log_qs = log_qs[0] if self.vectorize_particles else torch.stack(log_qs)
        log_weights = log_joints - log_qs.detach()

        # wake theta = iwae:
        log_sum_weight = torch.logsumexp(log_weights, dim=0)
        # TODO: check whether grad est should sum or mean over batch dim (currently sum)
        wake_theta_loss = -(log_sum_weight - math.log(self.num_particles)).sum()
        wake_theta_loss.backward(retain_graph=True)

        # wake phi = reweighted csis:
        normalised_weights = (log_weights - log_sum_weight).exp().detach()
        # TODO: check whether grad est should sum or mean over batch dim (currently sum)
        wake_phi_loss = -(normalised_weights * log_qs).sum(0).sum(0)
        wake_phi_loss.backward()

        # TODO: incorporate sleep/csis?

        warn_if_nan(wake_theta_loss, "loss")
        warn_if_nan(wake_phi_loss, "loss")
        return wake_theta_loss.detach(), wake_phi_loss.detach()
