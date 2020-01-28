# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import torch

import pyro
import pyro.poutine as poutine
from pyro.ops.stats import fit_generalized_pareto

from .abstract_infer import TracePosterior
from .enum import get_importance_trace


class Importance(TracePosterior):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    :param num_samples: number of samples to draw from the guide (default 10)

    This method performs posterior inference by importance sampling
    using the guide as the proposal distribution.
    If no guide is provided, it defaults to proposing from the model's prior.
    """

    def __init__(self, model, guide=None, num_samples=None):
        """
        Constructor. default to num_samples = 10, guide = model
        """
        super().__init__()
        if num_samples is None:
            num_samples = 10
            warnings.warn("num_samples not provided, defaulting to {}".format(num_samples))
        if guide is None:
            # propose from the prior by making a guide from the model by hiding observes
            guide = poutine.block(model, hide_types=["observe"])
        self.num_samples = num_samples
        self.model = model
        self.guide = guide

    def _traces(self, *args, **kwargs):
        """
        Generator of weighted samples from the proposal distribution.
        """
        for i in range(self.num_samples):
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, trace=guide_trace)).get_trace(*args, **kwargs)
            log_weight = model_trace.log_prob_sum() - guide_trace.log_prob_sum()
            yield (model_trace, log_weight)

    def get_log_normalizer(self):
        """
        Estimator of the normalizing constant of the target distribution.
        (mean of the unnormalized weights)
        """
        # ensure list is not empty
        if self.log_weights:
            log_w = torch.tensor(self.log_weights)
            log_num_samples = torch.log(torch.tensor(self.num_samples * 1.))
            return torch.logsumexp(log_w - log_num_samples, 0)
        else:
            warnings.warn("The log_weights list is empty, can not compute normalizing constant estimate.")

    def get_normalized_weights(self, log_scale=False):
        """
        Compute the normalized importance weights.
        """
        if self.log_weights:
            log_w = torch.tensor(self.log_weights)
            log_w_norm = log_w - torch.logsumexp(log_w, 0)
            return log_w_norm if log_scale else torch.exp(log_w_norm)
        else:
            warnings.warn("The log_weights list is empty. There is nothing to normalize.")

    def get_ESS(self):
        """
        Compute (Importance Sampling) Effective Sample Size (ESS).
        """
        if self.log_weights:
            log_w_norm = self.get_normalized_weights(log_scale=True)
            ess = torch.exp(-torch.logsumexp(2*log_w_norm, 0))
        else:
            warnings.warn("The log_weights list is empty, effective sample size is zero.")
            ess = 0
        return ess


def vectorized_importance_weights(model, guide, *args, **kwargs):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    :param num_samples: number of samples to draw from the guide (default 1)
    :param int max_plate_nesting: Bound on max number of nested :func:`pyro.plate` contexts.
    :param bool normalized: set to True to return self-normalized importance weights
    :returns: returns a ``(num_samples,)``-shaped tensor of importance weights
        and the model and guide traces that produced them

    Vectorized computation of importance weights for models with static structure::

        log_weights, model_trace, guide_trace = \\
            vectorized_importance_weights(model, guide, *args,
                                          num_samples=1000,
                                          max_plate_nesting=4,
                                          normalized=False)
    """
    num_samples = kwargs.pop("num_samples", 1)
    max_plate_nesting = kwargs.pop("max_plate_nesting", None)
    normalized = kwargs.pop("normalized", False)

    if max_plate_nesting is None:
        raise ValueError("must provide max_plate_nesting")
    max_plate_nesting += 1

    def vectorize(fn):
        def _fn(*args, **kwargs):
            with pyro.plate("num_particles_vectorized", num_samples, dim=-max_plate_nesting):
                return fn(*args, **kwargs)
        return _fn

    model_trace, guide_trace = get_importance_trace(
        "flat", max_plate_nesting, vectorize(model), vectorize(guide), args, kwargs)

    guide_trace.pack_tensors()
    model_trace.pack_tensors(guide_trace.plate_to_symbol)

    if num_samples == 1:
        log_weights = model_trace.log_prob_sum() - guide_trace.log_prob_sum()
    else:
        wd = guide_trace.plate_to_symbol["num_particles_vectorized"]
        log_weights = 0.
        for site in model_trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_weights += torch.einsum(site["packed"]["log_prob"]._pyro_dims + "->" + wd,
                                        [site["packed"]["log_prob"]])

        for site in guide_trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_weights -= torch.einsum(site["packed"]["log_prob"]._pyro_dims + "->" + wd,
                                        [site["packed"]["log_prob"]])

    if normalized:
        log_weights = log_weights - torch.logsumexp(log_weights)
    return log_weights, model_trace, guide_trace


@torch.no_grad()
def psis_diagnostic(model, guide, *args, **kwargs):
    """
    Computes the Pareto tail index k for a model/guide pair using the technique
    described in [1], which builds on previous work in [2]. If :math:`0 < k < 0.5`
    the guide is a good approximation to the model posterior, in the sense
    described in [1]. If :math:`0.5 \\le k \\le 0.7`, the guide provides a suboptimal
    approximation to the posterior, but may still be useful in practice. If
    :math:`k > 0.7` the guide program provides a poor approximation to the full
    posterior, and caution should be used when using the guide. Note, however,
    that a guide may be a poor fit to the full posterior while still yielding
    reasonable model predictions. If :math:`k < 0.0` the importance weights
    corresponding to the model and guide appear to be bounded from above; this
    would be a bizarre outcome for a guide trained via ELBO maximization. Please
    see [1] for a more complete discussion of how the tail index k should be
    interpreted.

    Please be advised that a large number of samples may be required for an
    accurate estimate of k.

    Note that we assume that the model and guide are both vectorized and have
    static structure. As is canonical in Pyro, the args and kwargs are passed
    to the model and guide.

    References
    [1] 'Yes, but Did It Work?: Evaluating Variational Inference.'
    Yuling Yao, Aki Vehtari, Daniel Simpson, Andrew Gelman
    [2] 'Pareto Smoothed Importance Sampling.'
    Aki Vehtari, Andrew Gelman, Jonah Gabry

    :param callable model: the model program.
    :param callable guide: the guide program.
    :param int num_particles: the total number of times we run the model and guide in
        order to compute the diagnostic. defaults to 1000.
    :param max_simultaneous_particles: the maximum number of simultaneous samples drawn
        from the model and guide. defaults to `num_particles`. `num_particles` must be
        divisible by `max_simultaneous_particles`. compute the diagnostic. defaults to 1000.
    :param int max_plate_nesting: optional bound on max number of nested :func:`pyro.plate`
        contexts in the model/guide. defaults to 7.
    :returns float: the PSIS diagnostic k
    """

    num_particles = kwargs.pop('num_particles', 1000)
    max_simultaneous_particles = kwargs.pop('max_simultaneous_particles', num_particles)
    max_plate_nesting = kwargs.pop('max_plate_nesting', 7)

    if num_particles % max_simultaneous_particles != 0:
        raise ValueError("num_particles must be divisible by max_simultaneous_particles.")

    N = num_particles // max_simultaneous_particles
    log_weights = [vectorized_importance_weights(model, guide, num_samples=max_simultaneous_particles,
                                                 max_plate_nesting=max_plate_nesting,
                                                 *args, **kwargs)[0] for _ in range(N)]
    log_weights = torch.cat(log_weights)
    log_weights -= log_weights.max()
    log_weights = torch.sort(log_weights, descending=False)[0]

    cutoff_index = - int(math.ceil(min(0.2 * num_particles, 3.0 * math.sqrt(num_particles)))) - 1
    lw_cutoff = max(math.log(1.0e-15), log_weights[cutoff_index])
    lw_tail = log_weights[log_weights > lw_cutoff]

    if len(lw_tail) < 10:
        warnings.warn("Not enough tail samples to compute PSIS diagnostic; increase num_particles.")
        k = float('inf')
    else:
        k, _ = fit_generalized_pareto(lw_tail.exp() - math.exp(lw_cutoff))

    return k
