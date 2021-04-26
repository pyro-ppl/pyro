# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Optional

import torch

import pyro
import pyro.poutine as poutine
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample


def is_latent(msg):
    if msg["type"] != "sample":
        return False
    if msg["is_observed"]:
        return False
    if site_is_subsample(msg):
        return False
    return True


class RequiresGradMessenger(Messenger):
    def __init__(self, predicate=lambda msg: True):
        self.predicate = predicate
        super().__init__()

    def _pyro_post_sample(self, msg):
        if is_latent(msg):
            if self.predicate(msg):
                msg["value"].requires_grad_()
            elif msg["value"].requires_grad:
                msg["value"] = msg["value"].detach()


def get_dependencies(
    model: Callable,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
    transitive: bool = False,
) -> Dict[str, List[str]]:
    """
    Infers direct dependencies among latent variables in a model.

    .. warning:: This currently relies on autograd and therefore works only for
        continuous random variables with differentiable dependencies.

    :param callable model: A model.
    :param tuple model_args: Optional tuple of model args.
    :param dict model_kwargs: Optional tuple of model args.
    :param bool transtitive: Whether to compute transitive dependencies (which
        is cheaper). Defaults to False, computing the finer direct dependencies.
    :returns: A dictionary whose keys are names of downstream latent sites
        and whose values are lists of names of upstream latent sites on which
        those downstream sites depend.
    :rtype: dict
    """
    if model_args is None:
        model_args = ()
    if model_kwargs is None:
        model_kwargs = {}

    def find_latents(predicate=lambda msg: True):
        with torch.enable_grad(), torch.random.fork_rng():
            with pyro.validation_enabled(False), RequiresGradMessenger(predicate):
                trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
        latents = [msg for msg in trace.nodes.values() if is_latent(msg)]
        return latents

    # First find transitive dependencies.
    latents = find_latents()
    dependencies = {msg["name"]: [] for msg in latents}
    for i, downstream in enumerate(latents):
        upstreams = latents[:i]
        if not upstreams:
            continue
        grads = torch.autograd.grad(
            downstream["fn"].log_prob(downstream["value"]).sum(),
            [u["value"] for u in upstreams],
            allow_unused=True,
        )
        for upstream, grad in zip(upstreams, grads):
            if grad is not None:
                dependencies[downstream["name"]].append(upstream["name"])
    if transitive:
        return dependencies

    # Then refine to direct dependencies.
    for i, downstream in enumerate(latents):
        for j, upstream in enumerate(latents[:max(0, i - 1)]):
            if upstream["name"] not in dependencies[downstream["name"]]:
                continue
            names = {upstream["name"], downstream["name"]}
            latents_ij = find_latents(lambda msg: msg["name"] in names)
            d = latents_ij[i]
            u = latents_ij[j]
            grad = torch.autograd.grad(
                d["fn"].log_prob(d["value"]).sum(),
                [u["value"]],
                allow_unused=True,
            )[0]
            if grad is None:
                dependencies[d["name"]].remove(u["name"])
    return dependencies


__all__ = [
    "get_dependencies",
]
