# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional

import torch

import pyro
import pyro.poutine as poutine
from pyro.poutine.trace_struct import Trace


class ResamplingCache:
    """Resampling cache for interactive tuning of generative models, typically
    when preforming prior predictive checks as an early step of Bayesian
    workflow.

    This is intended as a computational cache to speed up the interactive
    tuning of the parameters of prior distributions based on samples from a
    downstream simulation. The idea is that the simulation can be expensive,
    but that when one slightly tweaks parameters of the parameter distribution
    then one can reuse most of the previous samples via importance resampling.

    You can prewarm the cache by calling :meth:`sample` once with a dispersed
    ``prior`` and large ``num_samples``.

    :param callable model: A pyro model that takes no arguments.
    :param int max_plate_nesting: The maximum plate nesting in the model.
        If absent this will be guessed by running the model.
    """

    def __init__(self, model: callable, *, max_plate_nesting: Optional[int] = None):
        super().__init__()
        if max_plate_nesting is None:
            max_plate_nesting = _guess_max_plate_nesting(model)
        self.model = model
        self.max_plate_nesting = max_plate_nesting
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def sample(self, prior: callable, num_samples: int) -> Dict[str, torch.Tensor]:
        """Draws a weighted set of ``num_samples`` many model samples, each of
        which is conditioned on a sample from the prior.

        :param callable prior: A model with a subset of sites from ``self.model``.
        :param int num_samples: The number of samples to draw.
        :returns: A dictionary of stacked samples, with weights stored in the
            "_weights" key. Weights are all >= 1.
        :rtype: Dict[str, torch.Tensor]
        """
        samples: Dict[str, torch.Tensor] = {}

        # First try to reuse existing samples.
        if self._cache:
            # Importance sample: keep all weights >= 1; subsample weights < 1.
            batch_size = len(self._cache["_logp"])
            with poutine.block(), pyro.plate(
                "particles", batch_size, dim=-1 - self.max_plate_nesting
            ):
                trace = poutine.trace(prior).get_trace()
            new_logps = _log_prob_sum(trace, batch_size)
            old_logps = self._cache["_logp"]
            weights = (new_logps - old_logps).exp()
            u = self._cache["_u"]  # use memoized randomness to avoid twinkling
            accepted = (weights >= u).nonzero(as_tuple=True)[0][:num_samples]
            if len(accepted):
                samples["_weight"] = weights[accepted].clamp(min=1)
                for k, v in self._cache.items():
                    if not k.startswith("_"):
                        samples[k] = v[accepted]

        # Then possibly draw new samples.
        batch_size = num_samples - len(samples.get("_weight", ()))
        if batch_size > 0:
            # Draw samples from the prior model.
            with poutine.block(), pyro.plate(
                "particles", batch_size, dim=-1 - self.max_plate_nesting
            ):
                trace = poutine.trace(prior).get_trace()
                _extend(self._cache, "_logp", _log_prob_sum(trace, batch_size))
                _extend(self._cache, "_u", torch.rand(batch_size))
                _extend(samples, "_weight", torch.ones(batch_size))

                # Draw samples from the full model.
                trace = poutine.trace(poutine.replay(self.model, trace)).get_trace()
            for name, site in trace.nodes.items():
                if site["type"] == "sample" and not name.startswith("_"):
                    _extend(samples, name, site["value"])
                    _extend(self._cache, name, site["value"])

        assert len(set(map(len, self._cache.values()))) == 1
        assert all(len(v) == num_samples for v in samples.values())
        return samples


def _extend(destin: Dict[str, torch.Tensor], name: str, new: torch.Tensor) -> None:
    """Extens ``destin[name]`` tensor by a ``new`` tensor."""
    old = destin.get(name)
    destin[name] = new if old is None else torch.cat([old, new])


def _log_prob_sum(trace: Trace, batch_size: int) -> torch.Tensor:
    """Computes vectorized log_prob_sum batched over the leftmost dimension."""
    trace.compute_log_prob()
    result = 0.0
    for site in trace.nodes.values():
        if site["type"] == "sample":
            logp = site["log_prob"]
            print(f"DEBUG {site['name']} {tuple(logp.shape)}")
            assert logp.shape[:1] == (batch_size,)
            result += logp.reshape(batch_size, -1).sum(-1)
    return result


def _guess_max_plate_nesting(model: callable) -> int:
    with torch.no_grad(), poutine.block(), poutine.mask(mask=False):
        trace = poutine.trace(model).get_trace()
    plate_nesting = {0}.union(
        -f.dim
        for site in trace.nodes.values()
        for f in site.get("cond_indep_stack", [])
        if f.vectorized
    )
    return max(plate_nesting)
