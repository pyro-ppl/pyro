# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Optional

import torch

import pyro
import pyro.poutine as poutine
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample


def is_sample_site(msg):
    if msg["type"] != "sample":
        return False
    if site_is_subsample(msg):
        return False

    # Ignore masked observations.
    if msg["is_observed"] and msg["mask"] is False:
        return False

    # Exclude deterministic sites.
    fn = msg["fn"]
    while hasattr(fn, "base_dist"):
        fn = fn.base_dist
    if type(fn).__name__ == "Delta":
        return False

    return True


class RequiresGradMessenger(Messenger):
    def __init__(self, predicate=lambda msg: True):
        self.predicate = predicate
        super().__init__()

    def _pyro_post_sample(self, msg):
        if is_sample_site(msg):
            if self.predicate(msg):
                msg["value"].requires_grad_()
            elif not msg["is_observed"] and msg["value"].requires_grad:
                msg["value"] = msg["value"].detach()


def get_dependencies(
    model: Callable,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
) -> Dict[str, List[str]]:
    r"""
    Infers posterior dependencies among latent variables in a conditioned
    model.

    The resulting dependency graph can be treated as undirected, but is
    returned as a directed graph using the variable ordering in the model.

    .. warning:: This currently relies on autograd and therefore works only for
        continuous latent variables with differentiable dependencies. Discrete
        latent variables will raise errors. Gradient blocking may silently drop
        dependencies.

    **References**

    [1] S.Webb, A.Goliński, R.Zinkov, N.Siddharth, T.Rainforth, Y.W.Teh, F.Wood (2018)
        "Faithful inversion of generative models for effective amortized inference"
        https://dl.acm.org/doi/10.5555/3327144.3327229

    :param callable model: A model.
    :param tuple model_args: Optional tuple of model args.
    :param dict model_kwargs: Optional tuple of model args.
    :returns: A dictionary whose keys are names of downstream latent sites
        and whose values are lists of names of upstream latent sites on which
        those downstream sites depend.
    :rtype: dict
    """
    if model_args is None:
        model_args = ()
    if model_kwargs is None:
        model_kwargs = {}

    def get_sample_sites(predicate=lambda msg: True):
        with torch.enable_grad(), torch.random.fork_rng():
            with pyro.validation_enabled(False), RequiresGradMessenger(predicate):
                trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
        return [msg for msg in trace.nodes.values() if is_sample_site(msg)]

    # Collect observations.
    sample_sites = get_sample_sites()
    order = {msg["name"]: i for i, msg in enumerate(sample_sites)}
    observed = {msg["name"] for msg in sample_sites if msg["is_observed"]}

    # First find transitive dependencies among latent and observed sites
    prior_dependencies = {msg["name"]: set() for msg in sample_sites}
    for i, downstream in enumerate(sample_sites):
        upstreams = [u for u in sample_sites[:i] if not u["is_observed"]]
        if not upstreams:
            continue
        grads = torch.autograd.grad(
            downstream["fn"].log_prob(downstream["value"]).sum(),
            [u["value"] for u in upstreams],
            allow_unused=True,
        )
        for upstream, grad in zip(upstreams, grads):
            if grad is not None:
                prior_dependencies[downstream["name"]].add(upstream["name"])

    # Then refine to direct dependencies among latent and observed sites.
    for i, downstream in enumerate(sample_sites):
        for j, upstream in enumerate(sample_sites[:max(0, i - 1)]):
            if upstream["name"] not in prior_dependencies[downstream["name"]]:
                continue
            names = {upstream["name"], downstream["name"]}
            sample_sites_ij = get_sample_sites(lambda msg: msg["name"] in names)
            d = sample_sites_ij[i]
            u = sample_sites_ij[j]
            grad = torch.autograd.grad(
                d["fn"].log_prob(d["value"]).sum(),
                [u["value"]],
                allow_unused=True,
            )[0]
            if grad is None:
                prior_dependencies[d["name"]].remove(u["name"])

    # Next restrict to dependencies among latent variables.
    posterior_dependencies = {
        d: {u for u in upstreams if u not in observed}
        for d, upstreams in prior_dependencies.items()
        if d not in observed
    }

    # Finally add dependencies among latent variables in each Markov blanket.
    # This assumes all latents are eventually observed, at least indirectly.
    for d, upstreams in prior_dependencies.items():
        upstreams = {u for u in upstreams if u not in observed}
        for u1 in upstreams:
            for u2 in upstreams:
                if order[u1] < order[u2]:
                    posterior_dependencies[u2].add(u1)

    # Convert to a dict : str -> list.
    posterior_dependencies = {
        downstream: sorted(upstreams, key=order.__getitem__)
        for downstream, upstreams in posterior_dependencies.items()
    }
    return posterior_dependencies


__all__ = [
    "get_dependencies",
]
