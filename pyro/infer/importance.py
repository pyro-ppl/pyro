from __future__ import absolute_import, division, print_function
import torch
import warnings

import pyro
import pyro.poutine as poutine

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
        super(Importance, self).__init__()
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

    num_samples = kwargs.pop("num_samples", 1)
    max_plate_nesting = kwargs.pop("max_plate_nesting", 7)
    normalized = kwargs.pop("normalized", False)

    def vectorize(fn):
        def _fn(*args, **kwargs):
            with pyro.plate("num_particles_vectorized", num_samples, dim=-max_plate_nesting):
                return fn(*args, **kwargs)
        return _fn

    model_trace, guide_trace = get_importance_trace(
        "flat", max_plate_nesting, vectorize(model), vectorize(guide), *args, **kwargs)

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
                                        site["packed"]["log_prob"])

    if normalized:
        log_weights = log_weights - torch.logsumexp(log_weights)
    return log_weights, model_trace, guide_trace
