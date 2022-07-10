# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Callable, Dict, Iterator, List

import torch
from torch.autograd import grad
from torch.distributions.utils import lazy_property
from tqdm.auto import tqdm

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


class AnnealedMetropolisHastings(dist.Distribution):
    def __init__(
        self,
        model: Callable[[], Any],
        proposal: type,
        *,
        num_steps: int = 1000,
    ):
        super().__init__()
        self.model = model
        self.kernel = kernel

        # Use a 1/cosine schedule.
        t = torch.linspace(0, math.pi, 1 + num_steps)[1:]
        self._schedule = (2 / (1 - t.cos())).tolist()
        assert 10 < self._schedule[0] < math.inf
        assert 1 <= self._schedule[-1] <= 1.1

    def sample(self, shape=torch.Size()) -> Dict[str, torch.Tensor]:
        if shape:
            raise NotImplementedError
        for _, _, state in self._anneal(self.model):
            pass
        return state

    def log_prob(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        log_Z = self.log_partition_function
        trace = poutine.trace(poutine.condition(self.model, state)).get_trace()
        log_z = float(trace.log_prob_sum())
        return log_z - log_Z

    @lazy_property
    def log_partition_function(self) -> float:
        """
        Estimates the log partition function via thermodynamic annealing.
        """
        log_Z = 0.0
        old_temperature = 0.0
        with torch.no_grad(), poutine.block():
            # Draw samples from annealed models.
            for temperature, state in self._anneal(self.model):
                # Score each sample wrt the full temperature=1 model.
                trace = poutine.trace(poutine.condition(self.model, state)).get_trace()
                log_z = float(trace.log_prob_sum())
                log_Z += log_z * (temperature - old_temperature)
                old_temperature = temperature
        assert isinstance(log_Z, float)
        return log_Z

    def _anneal(self, model) -> Iterator[float, Dict[str, torch.Tensor]]:
        # Initialize arbitrarily to prior.
        trace = poutine.trace(model).get_trace()
        state = {k: v["value"] for k, v in trace.iter_stochastic_nodes()}

        for temperature in tqdm(self._schedule):
            scaled_model = poutine.scale(model, scale=1 / temperature)
            state = kernel(scaled_model, state)
            yield temperature, state


def batched_log_prob_sum(
    model: callable,
    state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    result = 0.0
    with torch.no_grad(), poutine.block():
        trace = poutine.trace(poutine.condition(model, state)).get_trace()
        for site in trace.nodes.values():
            if site["type"] == "sample":
                logp = site["log_prob"]
                sum_dims = [f.dim for f in site["cond_indep_stack"]]
                if sum_dims:
                    logp = logp.sum(sum_dims, keepdim=True)
                result += logp
    return result


class DiscreteLangevinProposal:
    def __init__(self, model, state):
        state_grad = {k: v.clone().requires_grad_() for k, v in state.items()}
        trace = poutine.trace(poutine.condition(model, state_grad)).get_trace()
        logp = trace.log_prob_sum()

        # Compute proposal distribution.
        nbhd = dict(zip(state_grad, grad(logp, list(state_grad.values()))))
        logq = {}
        for k, x in state.items():
            v = nbhd[k].detach()
            v -= (v * x).sum(-1, True)
            logq[k] = v / 2
            logq[k]

        self.x = state
        self.logp = logp
        self.logq = logq

    def sample(self):
        x = 
