# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.distributions import Distribution


class ResamplingCache:
    """Resampling cache for interactive tuning of distributions, typically when
    preforming prior predictive checks as an early step of Bayesian workflow.

    This is intended as a computational cache to speed up the interactive
    tuning of the parameters of a distribution based on samples from a
    downstream simulation. The idea is that the simulation can be expensive,
    but that when one slightly tweaks parameters of the parameter distribution
    then one can reuse most of the previous samples via importance resampling.

    You can prewarm the cache by calling :meth:`sample` once with a dispersed
    ``distribution`` and large ``num_samples``.

    :param callable model: A simulator that inputs a single parameter and
        outputs some data.
    :param int batch_size: Optional batch size if the model can be run in
        batch.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], Any],
        batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.cache: List[Tuple[float, torch.Tensor, Any, float]] = []

    def sample(self, distribution: Distribution, num_samples: int) -> list:
        """Draws a list of at least ``num_samples`` many model samples, each of
        whose model inputs is drawn from the given distribution.

        :param Distribution distribution: A distribution object.
        :param int num_samples: The number of samples to draw.
        :returns: A list of (weight, sample) pairs, where each sample is
             the result of a simulation.
        :rtype: list
        """
        weighted_samples = self.weighted_sample(distribution, num_samples)
        samples = []

        for weight, sample in weighted_samples:
            # Replicate E[weight]-many times.
            if weight == 1:
                samples.append(sample)
            else:
                samples.extend([sample] * int(weight))
                if weight % 1 > torch.rand(()):
                    samples.append(sample)

        assert len(samples) >= num_samples
        return samples

    def weighted_sample(
        self, distribution: Distribution, num_samples: int
    ) -> List[Tuple[float, Any]]:
        """Draws a weighted set of ``num_samples`` many model samples, each of
        whose model inputs is drawn from the given distribution.

        :param Distribution distribution: A distribution object.
        :param int num_samples: The number of samples to draw.
        :returns: A list of (weight, sample) pairs, where each sample is
             the result of a simulation. Weights are always >= 1.
        :rtype: List[Tuple[float, Any]]
        """
        weighted_samples = []

        # First try to reuse existing samples.
        if self.cache:
            old_logps = torch.tensor([row[0] for row in self.cache])
            params = torch.stack([row[1] for row in self.cache])
            us = torch.tensor([row[3] for row in self.cache])
            new_logps = distribution.log_prob(params)
            # Importance sample: keep all weights > 1, and subsample weights < 1.
            weights = (new_logps - old_logps).exp()
            accepted = (weights > us).nonzero(as_tuple=True)[0].tolist()
            weights.clamp_(min=1)
            for i in accepted[:num_samples]:
                weight = float(weights[i])
                sample = self.cache[i][2]
                weighted_samples.append((weight, sample))

        # Then possibly draw new samples.
        while len(weighted_samples) < num_samples:
            if self.batch_size is None:
                params = [distribution.sample()]
                log_probs = [distribution.log_prob(params[0])]
                samples = [self.model(params[0])]
            else:
                batch_size = min(self.batch_size, num_samples - len(weighted_samples))
                params = distribution.sample([batch_size])
                log_probs = distribution.log_prob(params)
                samples = self.model(params)

            for log_prob, param, sample in zip(log_probs, params, samples):
                u = float(torch.rand(()))  # save randomness to avoid twinkling
                self.cache.append((log_prob, param, sample, u))
                weighted_samples.append((1.0, sample))

        assert len(weighted_samples) == num_samples
        return weighted_samples
