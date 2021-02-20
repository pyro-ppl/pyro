# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import queue
import warnings

import torch

import pyro.poutine as poutine

from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace, iter_discrete_escape, iter_discrete_extend
from pyro.infer.util import compute_site_dice_factor, is_validation_enabled, torch_item
from pyro.ops import packed
from pyro.ops.contract import einsum
from pyro.poutine.enum_messenger import EnumMessenger
from pyro.util import check_traceenum_requirements, warn_if_nan


def _compute_dice_factors(model_trace, guide_trace):
    """
    compute per-site DiCE log-factors for non-reparameterized proposal sites
    this logic is adapted from pyro.infer.util.Dice.__init__
    """
    log_probs = []
    for role, trace in zip(("model", "guide"), (model_trace, guide_trace)):
        for name, site in trace.nodes.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            if role == "model" and name in guide_trace:
                continue

            log_prob, log_denom = compute_site_dice_factor(site)
            if not is_identically_zero(log_denom):
                dims = log_prob._pyro_dims
                log_prob = log_prob - log_denom
                log_prob._pyro_dims = dims
            if not is_identically_zero(log_prob):
                log_probs.append(log_prob)

    return log_probs


def _compute_tmc_factors(model_trace, guide_trace):
    """
    compute per-site log-factors for all observed and unobserved variables
    log-factors are log(p / q) for unobserved sites and log(p) for observed sites
    """
    log_factors = []
    for name, site in guide_trace.nodes.items():
        if site["type"] != "sample" or site["is_observed"]:
            continue
        log_proposal = site["packed"]["log_prob"]
        log_factors.append(packed.neg(log_proposal))
    for name, site in model_trace.nodes.items():
        if site["type"] != "sample":
            continue
        if site["name"] not in guide_trace and \
                not site["is_observed"] and \
                site["infer"].get("enumerate", None) == "parallel" and \
                site["infer"].get("num_samples", -1) > 0:
            # site was sampled from the prior
            log_proposal = packed.neg(site["packed"]["log_prob"])
            log_factors.append(log_proposal)
        log_factors.append(site["packed"]["log_prob"])
    return log_factors


def _compute_tmc_estimate(model_trace, guide_trace):
    """
    Use :func:`~pyro.ops.contract.einsum` to compute the Tensor Monte Carlo
    estimate of the marginal likelihood given parallel-sampled traces.
    """
    # factors
    log_factors = _compute_tmc_factors(model_trace, guide_trace)
    log_factors += _compute_dice_factors(model_trace, guide_trace)

    if not log_factors:
        return 0.

    # loss
    eqn = ",".join([f._pyro_dims for f in log_factors]) + "->"
    plates = "".join(frozenset().union(list(model_trace.plate_to_symbol.values()),
                                       list(guide_trace.plate_to_symbol.values())))
    tmc, = einsum(eqn, *log_factors, plates=plates,
                  backend="pyro.ops.einsum.torch_log",
                  modulo_total=False)
    return tmc


class TraceTMC_ELBO(ELBO):
    """
    A trace-based implementation of Tensor Monte Carlo [1]
    by way of Tensor Variable Elimination [2] that supports:
    - local parallel sampling over any sample site in the model or guide
    - exhaustive enumeration over any sample site in the model or guide

    To take multiple samples, mark the site with
    ``infer={'enumerate': 'parallel', 'num_samples': N}``.
    To configure all sites in a model or guide at once,
    use :func:`~pyro.infer.enum.config_enumerate` .
    To enumerate or sample a sample site in the ``model``,
    mark the site and ensure the site does not appear in the ``guide``.

    This assumes restricted dependency structure on the model and guide:
    variables outside of an :class:`~pyro.plate` can never depend on
    variables inside that :class:`~pyro.plate` .

    References

    [1] `Tensor Monte Carlo: Particle Methods for the GPU Era`,
        Laurence Aitchison (2018)

    [2] `Tensor Variable Elimination for Plated Factor Graphs`,
        Fritz Obermeyer, Eli Bingham, Martin Jankowiak, Justin Chiu, Neeraj Pradhan,
        Alexander Rush, Noah Goodman (2019)
    """

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs)

        if is_validation_enabled():
            check_traceenum_requirements(model_trace, guide_trace)

            has_enumerated_sites = any(site["infer"].get("enumerate")
                                       for trace in (guide_trace, model_trace)
                                       for name, site in trace.nodes.items()
                                       if site["type"] == "sample")

            if self.strict_enumeration_warning and not has_enumerated_sites:
                warnings.warn('Found no sample sites configured for enumeration. '
                              'If you want to enumerate sites, you need to @config_enumerate or set '
                              'infer={"enumerate": "sequential"} or infer={"enumerate": "parallel"}? '
                              'If you do not want to enumerate, consider using Trace_ELBO instead.')

        model_trace.compute_score_parts()
        guide_trace.pack_tensors()
        model_trace.pack_tensors(guide_trace.plate_to_symbol)
        return model_trace, guide_trace

    def _get_traces(self, model, guide, args, kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.max_plate_nesting == float('inf'):
            self._guess_max_plate_nesting(model, guide, args, kwargs)
        if self.vectorize_particles:
            guide = self._vectorized_num_particles(guide)
            model = self._vectorized_num_particles(model)

        # Enable parallel enumeration over the vectorized guide and model.
        # The model allocates enumeration dimensions after (to the left of) the guide,
        # accomplished by preserving the _ENUM_ALLOCATOR state after the guide call.
        guide_enum = EnumMessenger(first_available_dim=-1 - self.max_plate_nesting)
        model_enum = EnumMessenger()  # preserve _ENUM_ALLOCATOR state
        guide = guide_enum(guide)
        model = model_enum(model)

        q = queue.LifoQueue()
        guide = poutine.queue(guide, q,
                              escape_fn=iter_discrete_escape,
                              extend_fn=iter_discrete_extend)
        for i in range(1 if self.vectorize_particles else self.num_particles):
            q.put(poutine.Trace())
            while not q.empty():
                yield self._get_trace(model, guide, args, kwargs)

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        :returns: a differentiable estimate of the marginal log-likelihood
        :rtype: torch.Tensor
        :raises ValueError: if the ELBO is not differentiable (e.g. is
            identically zero)

        Computes a differentiable TMC estimate using ``num_particles`` many samples
        (particles).  The result should be infinitely differentiable (as long
        as underlying derivatives have been implemented).
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = _compute_tmc_estimate(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo = elbo + elbo_particle
        elbo = elbo / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, guide, *args, **kwargs):
        with torch.no_grad():
            loss = self.differentiable_loss(model, guide, *args, **kwargs)
            if is_identically_zero(loss) or not loss.requires_grad:
                return torch_item(loss)
            return loss.item()

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        if is_identically_zero(loss) or not loss.requires_grad:
            return torch_item(loss)
        loss.backward()
        return loss.item()
