# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional

import torch

import pyro
import pyro.poutine as poutine
from pyro.poutine.trace_struct import Trace


class Resampler:
    """Resampler for interactive tuning of generative models, typically
    when preforming prior predictive checks as an early step of Bayesian
    workflow.

    This is intended as a computational cache to speed up the interactive
    tuning of the parameters of prior distributions based on samples from a
    downstream simulation. The idea is that the simulation can be expensive,
    but that when one slightly tweaks parameters of the parameter distribution
    then one can reuse most of the previous samples via importance resampling.

    :param callable model: A pyro model that takes no arguments.
    :param callable guide: An initial guide over a subset of the model's
        latent variables. The initial guide should be diffuse, covering
        the subsequent guides passed to :meth:`sample`.
    :param int num_samples: Number of inital samples to draw. This should
        be much larger than the ``num_samples`` requested in subsequent
        calls to :meth:`sample`.
    :param int max_plate_nesting: The maximum plate nesting in the model.
        If absent this will be guessed by running the model.
    """

    def __init__(
        self,
        model: Callable,
        guide: Callable,
        num_samples: int,
        *,
        max_plate_nesting: Optional[int] = None,
    ):
        super().__init__()
        if max_plate_nesting is None:
            max_plate_nesting = _guess_max_plate_nesting(model)
        self._particle_dim = -1 - max_plate_nesting

        # Draw samples from the initial guide.
        with pyro.plate("particles", num_samples, dim=self._particle_dim):
            trace = poutine.trace(guide).get_trace()
            print(f"DEBUG1 {trace.nodes['alpha']['value'].mean(0)}")
            self._old_logp = _log_prob_sum(trace, num_samples)

            # Draw samples from the full model.
            trace = poutine.trace(poutine.replay(model, trace)).get_trace()
        self._samples = {
            name: site["value"]
            for name, site in trace.nodes.items()
            if site["type"] == "sample"
        }
        print(f"DEBUG2 {self._samples['alpha'].mean(0)}")

    @torch.no_grad()
    def sample(self, guide: Callable, num_samples: int) -> Dict[str, torch.Tensor]:
        """Draws a set of at most ``num_samples`` many model samples, each of
        which is conditioned on a sample from the given ``guide``.

        Internally this importance resamples the samples generated in
        ``.__init__()``, and does not rerun the ``model``. If the original
        samples poorly cover the returned samples will show low diversity.

        :param callable guide: A model with a subset of sites from ``model``.
        :param int num_samples: The number of samples to draw.
        :returns: A dictionary of stacked samples.
        :rtype: Dict[str, torch.Tensor]
        """
        # Importance sample: keep all weights >= 1; subsample weights < 1.
        batch_size = len(self._old_logp)
        with pyro.plate("particles", batch_size, dim=self._particle_dim):
            trace = poutine.trace(guide).get_trace()
            print(f"DEBUG3 {trace.nodes['alpha']['value'].mean(0)}")
        new_logp = _log_prob_sum(trace, batch_size)
        weights = (new_logp - self._old_logp).exp()
        i = torch.multinomial(weights, num_samples, replacement=True)
        samples = {k: v[i] for k, v in self._samples.items()}
        print(f"DEBUG4 {samples['alpha'].mean(0)}")
        return samples


def _log_prob_sum(trace: Trace, batch_size: int) -> torch.Tensor:
    """Computes vectorized log_prob_sum batched over the leftmost dimension."""
    trace.compute_log_prob()
    result = 0.0
    for site in trace.nodes.values():
        if site["type"] == "sample":
            logp = site["log_prob"]
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
