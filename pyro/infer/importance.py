from __future__ import absolute_import, division, print_function

import logging

import pyro.poutine as poutine

from .abstract_infer import TracePosterior

logger = logging.getLogger(__name__)


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
            logger.warn("num_samples not provided, defaulting to {}".format(num_samples))
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
