# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch.distributions import Distribution


class _CacheRow(NamedTuple):
    u: float
    logp: float
    param: Dict[str, torch.Tensor]
    sample: Any


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

    def __init__(self, model: callable, batch_size: Optional[int] = None):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.clear()

    def clear(self) -> None:
        """Clears the cache."""
        # This cache is the source of truth.
        self.cache: List[_CacheRow] = []
        # These temporary tensors are used only for speed.
        self._us: Optional[torch.Tensor] = None
        self._logps: Optional[torch.Tensor] = None
        self._params: Optional[Dict[str, torch.Tensor]] = None
        # This memoized noise reduces twinkling.
        self._vs: Dict[int, float] = defaultdict(lambda: torch.randn(()).item())

    def sample(self, prior: Dict[str, Distribution], num_samples: int) -> list:
        """Draws a list of at least ``num_samples`` many model outputs, each of
        whose model kwargs is drawn from the given prior.

        This is an efficient cached sampler roughly equivalent to::

            samples = [
                model(**{k: v.sample() for k, v in prior.items()})
                for _ in range(num_samples)
            ]

        :param dict prior: A dictionary mapping names to distribution objects.
            The model should take the prior's keys as kwargs.
        :param int num_samples: The number of samples to draw.
        :returns: A list of (weight, sample) pairs, where each sample is
             the result of a simulation.
        :rtype: list
        """

        weighted_samples = self.weighted_sample(prior, num_samples)
        samples = []

        for weight, sample in weighted_samples:
            # Replicate E[weight]-many times.
            if weight == 1:
                samples.append(sample)
            else:
                samples.extend([sample] * int(weight))
                unif01 = self._vs[id(sample)]
                if weight % 1 > unif01:
                    samples.append(sample)

        assert len(samples) >= num_samples
        return samples

    def weighted_sample(
        self, prior: Dict[str, Distribution], num_samples: int
    ) -> List[Tuple[float, Any]]:
        """Draws a weighted set of ``num_samples`` many model outputs, each of
        whose model kwargs are drawn from the given prior.

        :param dict prior: A dictionary mapping names to distribution objects.
            The model should take the prior's keys as kwargs.
        :param int num_samples: The number of samples to draw.
        :returns: A list of (weight, sample) pairs, where each sample is
             the result of a simulation. Weights are always >= 1.
        :rtype: List[Tuple[float, Any]]
        """
        weighted_samples = []
        prior_ = _DistributionDict(prior)

        # First try to reuse existing samples.
        if self.cache:
            # Importance sample: keep all weights > 1, and subsample weights < 1.
            us, old_logps, old_param, old_samples = self._read_cache()
            new_logps = prior_.log_prob(old_param)
            weights = (new_logps - old_logps).exp()
            # we memoize randomness to avoid twinkling
            accepted = (weights > us).nonzero(as_tuple=True)[0]
            weights.clamp_(min=1)
            for i in accepted[:num_samples].tolist():
                weight = float(weights[i])
                sample = self.cache[i].sample
                weighted_samples.append((weight, sample))

        # Then possibly draw new samples.
        while len(weighted_samples) < num_samples:
            if self.batch_size is None:
                param = prior_.sample()
                params = {k: [v] for k, v in param.items()}
                log_probs = [prior_.log_prob(param)]
                samples = [self.model(**param)]
                us = torch.rand((1,))
            else:
                batch_size = min(self.batch_size, num_samples - len(weighted_samples))
                params = prior_.sample([batch_size])
                log_probs = prior_.log_prob(params)
                samples = self.model(**params)
                us = torch.rand((batch_size,))

            for i, (u, log_prob, sample) in enumerate(zip(us, log_probs, samples)):
                param = {k: v[i] for k, v in params.items()}
                self.cache.append(_CacheRow(float(u), float(log_prob), param, sample))
                weighted_samples.append((1.0, sample))

        assert len(weighted_samples) == num_samples
        return weighted_samples

    def _read_cache(self):
        assert self.cache
        if self._us is None:
            # Initialize tensors.
            self._us = torch.tensor([row.u for row in self.cache])
            self._logps = torch.tensor([row.logp for row in self.cache])
            self._params = {
                k: torch.stack([row.param[k] for row in self.cache])
                for k in self.cache[0].param
            }
        elif len(self.cache) > len(self._logps):
            # Extend tensors.
            added = self.cache[len(self._logps) :]
            self._us = torch.cat([self._us, torch.tensor([row[0] for row in added])])
            self._logps = torch.cat(
                [self._logps, torch.tensor([row[1] for row in added])]
            )
            self._params = {
                k: torch.cat([v, torch.stack([row[2][k] for row in added])])
                for k, v in self._params.items()
            }

        samples = [row.sample for row in self.cache]
        return self._us, self._logps, self._params, samples


class _DistributionDict:
    """Helper to operate on a dictionary of distributions."""

    def __init__(self, dists: Dict[str, Distribution]):
        self.dists = dists

    def sample(self, sample_shape=torch.Size()) -> Dict[str, torch.Tensor]:
        return {k: v.sample(sample_shape) for k, v in self.dists.items()}

    def log_prob(self, values: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert set(self.dists) == set(values)
        return sum(v.log_prob(values[k]) for k, v in self.dists.items())
