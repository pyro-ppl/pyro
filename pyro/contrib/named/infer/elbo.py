# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Tuple

import torch
from functorch.dim import Dim
from typing_extensions import ParamSpec

import pyro
from pyro import poutine
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.infer import ELBO as _OrigELBO
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import Message

_P = ParamSpec("_P")


class ELBO(_OrigELBO):
    def _get_trace(self, *args, **kwargs):
        raise RuntimeError("shouldn't be here!")

    def differentiable_loss(self, model, guide, *args, **kwargs):
        raise NotImplementedError("Must implement differentiable_loss")

    def loss(self, model, guide, *args, **kwargs):
        return self.differentiable_loss(model, guide, *args, **kwargs).detach().item()

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        loss.backward()
        return loss.item()


def track_provenance(x: torch.Tensor, provenance: Dim) -> torch.Tensor:
    return x.unsqueeze(0)[provenance]


class track_nonreparam(Messenger):
    def _pyro_post_sample(self, msg: Message) -> None:
        if (
            msg["type"] == "sample"
            and isinstance(msg["fn"], TorchDistributionMixin)
            and not msg["is_observed"]
            and not msg["fn"].has_rsample
        ):
            provenance = Dim(msg["name"])
            msg["value"] = track_provenance(msg["value"], provenance)


def get_importance_trace(
    model: Callable[_P, Any],
    guide: Callable[_P, Any],
    *args: _P.args,
    **kwargs: _P.kwargs
) -> Tuple[poutine.Trace, poutine.Trace]:
    """
    Returns traces from the guide and the model that is run against it.
    The returned traces also store the log probability at each site.
    """
    with track_nonreparam():
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    replay_model = poutine.replay(model, trace=guide_trace)
    model_trace = poutine.trace(replay_model).get_trace(*args, **kwargs)

    for is_guide, trace in zip((True, False), (guide_trace, model_trace)):
        for site in list(trace.nodes.values()):
            if site["type"] == "sample" and isinstance(
                site["fn"], TorchDistributionMixin
            ):
                log_prob = site["fn"].log_prob(site["value"])
                site["log_prob"] = log_prob

                if is_guide and not site["fn"].has_rsample:
                    # importance sampling weights
                    site["log_measure"] = log_prob - log_prob.detach()
            else:
                trace.remove_node(site["name"])
    return model_trace, guide_trace


class Trace_ELBO(ELBO):
    def differentiable_loss(
        self,
        model: Callable[_P, Any],
        guide: Callable[_P, Any],
        *args: _P.args,
        **kwargs: _P.kwargs
    ) -> torch.Tensor:
        if self.num_particles > 1:
            vectorize = pyro.plate(
                "num_particles", self.num_particles, dim=Dim("num_particles")
            )
            model = vectorize(model)
            guide = vectorize(guide)

        model_trace, guide_trace = get_importance_trace(model, guide, *args, **kwargs)

        cost_terms = []
        # logp terms
        for site in model_trace.nodes.values():
            cost = site["log_prob"]
            scale = site["scale"]
            batch_dims = tuple(f.dim for f in site["cond_indep_stack"])
            deps = tuple(set(getattr(cost, "dims", ())) - set(batch_dims))
            cost_terms.append((cost, scale, batch_dims, deps))
        # -logq terms
        for site in guide_trace.nodes.values():
            cost = -site["log_prob"]
            scale = site["scale"]
            batch_dims = tuple(f.dim for f in site["cond_indep_stack"])
            deps = tuple(set(getattr(cost, "dims", ())) - set(batch_dims))
            cost_terms.append((cost, scale, batch_dims, deps))

        elbo = 0.0
        for cost, scale, batch_dims, deps in cost_terms:
            if deps:
                dice_factor = 0.0
                for key in deps:
                    dice_factor += guide_trace.nodes[str(key)]["log_measure"]
                dice_factor_dims = getattr(dice_factor, "dims", ())
                cost_dims = getattr(cost, "dims", ())
                sum_dims = tuple(set(dice_factor_dims) - set(cost_dims))
                if sum_dims:
                    dice_factor = dice_factor.sum(sum_dims)
                cost = torch.exp(dice_factor) * cost
                cost = cost.mean(deps)
            if scale is not None:
                cost = cost * scale
            elbo += cost.sum(batch_dims) / self.num_particles

        return -elbo
