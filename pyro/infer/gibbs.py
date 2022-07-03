# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, Callable, Dict, List

import torch
from tqdm.auto import tqdm

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


class Kernel(ABC):
    @abstractmethod
    def sample(
        self,
        model: Callable[[], Any],
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class AnnealedMCMC(dist.Distribution):
    def __init__(
        self,
        model: Callable[[], Any],
        kernel: Callable[[callable, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        *,
        num_steps: int = 10000,
        max_plate_nesting: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.kernel = kernel
        self.max_plate_nesting = max_plate_nesting

        # Use a cosine schedule.
        t = torch.linspace(0, math.pi, 1 + num_steps)[1:]
        self._schedule = ((1 - t.cos()) / 2).tolist()
        assert 0 <= self._schedule[0] < 0.1
        assert 0.9 < self._schedule[-1] <= 1

    @lazy_property
    def log_partition_function(self) -> float:
        log_Z = 0.0
        old_temperature = 0.0
        with torch.no_grad(), poutine.block():
            # Draw samples from annealed models.
            for temperature, model, state in self._anneal(self.model):
                # Score each sample wrt the full temperature=1 model.
                trace = poutine.trace(poutine.condition(self.model, state)).get_trace()
                log_z = float(trace.log_prob_sum())
                log_Z += log_z * (temperature - old_temperature)
                old_temperature = temperature
        assert isinstance(log_Z, float)
        return log_Z

    def sample(self, shape=torch.Size()) -> Dict[str, torch.Tensor]:
        shape = torch.Size(shape)
        if shape.numel() == 1:
            model = self.model
        else:
            vectorize = pyro.plate(
                "particles", shape.numel(), dim=-1 - self.max_plate_nesting
            )
            model = vectorize(self.model)

        for _, _, state in self._anneal():
            pass

        return state

    def log_prob(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        log_Z = self.log_partition_function
        return batched_log_prob(self.model, state) - log_Z

    def _anneal(self, model):
        # Initialize arbitrarily to prior.
        trace = poutine.trace(model).get_trace()
        state = {k: v["value"] for k, v in trace.iter_stochastic_nodes()}

        for temperature in tqdm(self._schedule):
            scaled_model = poutine.scale(model, scale=1 / temperature)
            state = kernel(scaled_model, state)
            yield temperature, scaled_model, state


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


class SingleSiteGibbsKernel(Kernel):
    pass


class GibbsWithGradientsKernel(Kernel):
    pass
