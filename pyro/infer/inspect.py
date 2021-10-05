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
        if is_sample_site(msg) and msg["value"].dtype.is_floating_point:
            if self.predicate(msg):
                if not msg["value"].requires_grad:
                    msg["value"] = msg["value"][...].requires_grad_()
            elif not msg["is_observed"] and msg["value"].requires_grad:
                if msg["value"].requires_grad:
                    msg["value"] = msg["value"].detach()


@torch.enable_grad()
def get_dependencies(
    model: Callable,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    r"""
    EXPERIMENTAL Infers dependency structure about a conditioned model.

    This returns a nested dictionary with structure like::

        {
            "prior_dependencies": {
                "variable1": {"variable1": set()},
                "variable2": {"variable1": set(), "variable2": set()},
                ...
            },
            "posterior_dependencies": {
                "variable1": {"variable1": {"plate1"}, "variable2": set()},
                ...
            },
        }

    where

    -   `prior_dependencies` is a dict mapping downstream latent and observed
        variables to dictionaries mapping upstream latent variables on which
        they depend to sets of plates inducing full dependencies.
        That is, included plates introduce quadratically many dependencies as
        in complete-bipartite graphs, whereas excluded plates introduce only
        linearly many dependencies as in independent sets of parallel edges.
        Prior dependencies follow the original model order.
    -   `posterior_dependencies` is a similar dict, but mapping latent
        variables to the latent or observed sits on which they depend in the
        posterior. Posterior dependencies are reversed from the model order.

    Dependencies elide ``pyro.deterministic`` sites and ``pyro.sample(...,
    Delta(...))`` sites.

    **Examples**

    Here is a simple example with no plates. We see every node depends on
    itself, and only the latent variables appear in the posterior::

        def model_1():
            a = pyro.sample("a", dist.Normal(0, 1))
            pyro.sample("b", dist.Normal(a, 1), obs=torch.tensor(0.0))

        assert get_dependencies(model_1) == {
            "prior_dependencies": {
                "a": {"a": set()},
                "b": {"a": set(), "b": set()},
            },
            "posterior_dependencies": {
                "a": {"a": set(), "b": set()},
            },
        }

    Here is an example where two variables ``a`` and ``b`` start out
    conditionally independent in the prior, but become conditionally dependent
    in the posterior do the so-called collider variable ``c`` on which they
    both depend. This is called "moralization" in the graphical model
    literature::

        def model_2():
            a = pyro.sample("a", dist.Normal(0, 1))
            b = pyro.sample("b", dist.LogNormal(0, 1))
            c = pyro.sample("c", dist.Normal(a, b))
            pyro.sample("d", dist.Normal(c, 1), obs=torch.tensor(0.))

        assert get_dependencies(model_2) == {
            "prior_dependencies": {
                "a": {"a": set()},
                "b": {"b": set()},
                "c": {"a": set(), "b": set(), "c": set()},
                "d": {"c": set(), "d": set()},
            },
            "posterior_dependencies": {
                "a": {"a": set(), "b": set(), "c": set()},
                "b": {"b": set(), "c": set()},
                "c": {"c": set(), "d": set()},
            },
        }

    Dependencies can be more complex in the presence of plates. So far all the
    dict values have been empty sets of plates, but in the following posterior
    we see that ``c`` depends on itself across the plate ``p``. This means
    that, among the elements of ``c``, e.g. ``c[0]`` depends on ``c[1]`` (this
    is why we explicitly allow variables to depend on themselves)::

        def model_3():
            with pyro.plate("p", 5):
                a = pyro.sample("a", dist.Normal(0, 1))
            pyro.sample("b", dist.Normal(a.sum(), 1), obs=torch.tensor(0.0))

        assert get_dependencies(model_3) == {
            "prior_dependencies": {
                "a": {"a": set()},
                "b": {"a": set(), "b": set()},
            },
            "posterior_dependencies": {
                "a": {"a": {"p"}, "b": set()},
            },
        }

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
    :param dict model_kwargs: Optional dict of model kwargs.
    :returns: A dictionary of metadata (see above).
    :rtype: dict
    """
    if model_args is None:
        model_args = ()
    if model_kwargs is None:
        model_kwargs = {}

    def get_sample_sites(predicate=lambda msg: True):
        with torch.random.fork_rng():
            with pyro.validation_enabled(False), RequiresGradMessenger(predicate):
                trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
        return [msg for msg in trace.nodes.values() if is_sample_site(msg)]

    # Collect observations.
    sample_sites = get_sample_sites()
    observed = {msg["name"] for msg in sample_sites if msg["is_observed"]}
    plates = {
        msg["name"]: {f.name for f in msg["cond_indep_stack"] if f.vectorized}
        for msg in sample_sites
    }

    # First find transitive dependencies among latent and observed sites
    prior_dependencies = {n: {n: set()} for n in plates}  # no deps yet
    for i, downstream in enumerate(sample_sites):
        upstreams = [
            u for u in sample_sites[:i] if not u["is_observed"] if u["value"].numel()
        ]
        if not upstreams:
            continue
        log_prob = downstream["fn"].log_prob(downstream["value"]).sum()
        grads = _safe_grad(log_prob, [u["value"] for u in upstreams])
        for upstream, grad in zip(upstreams, grads):
            if grad is not None:
                d = downstream["name"]
                u = upstream["name"]
                prior_dependencies[d][u] = set()

    # Then refine to direct dependencies among latent and observed sites.
    for i, downstream in enumerate(sample_sites):
        for j, upstream in enumerate(sample_sites[: max(0, i - 1)]):
            if upstream["name"] not in prior_dependencies[downstream["name"]]:
                continue
            names = {upstream["name"], downstream["name"]}
            sample_sites_ij = get_sample_sites(lambda msg: msg["name"] in names)
            d = sample_sites_ij[i]
            u = sample_sites_ij[j]
            log_prob = d["fn"].log_prob(d["value"]).sum()
            grad = _safe_grad(log_prob, [u["value"]])[0]
            if grad is None:
                prior_dependencies[d["name"]].pop(u["name"])

    # Next reverse dependencies and restrict downstream nodes to latent sites.
    posterior_dependencies = {n: {} for n in plates if n not in observed}
    for d, upstreams in prior_dependencies.items():
        for u, p in upstreams.items():
            if u not in observed:
                # Note the folowing reverses:
                # u is henceforth downstream and d is henceforth upstream.
                posterior_dependencies[u][d] = p.copy()

    # Moralize: add dependencies among latent variables in each Markov blanket.
    # This assumes all latents are eventually observed, at least indirectly.
    order = {msg["name"]: i for i, msg in enumerate(reversed(sample_sites))}
    for d, upstreams in prior_dependencies.items():
        upstreams = {u: p for u, p in upstreams.items() if u not in observed}
        for u1, p1 in upstreams.items():
            for u2, p2 in upstreams.items():
                if order[u1] <= order[u2]:
                    p12 = posterior_dependencies[u2].setdefault(u1, set())
                    p12 |= plates[u1] & plates[u2] - plates[d]
                    p12 |= plates[u2] & p1
                    p12 |= plates[u1] & p2

    return {
        "prior_dependencies": prior_dependencies,
        "posterior_dependencies": posterior_dependencies,
    }


def _safe_grad(root: torch.Tensor, args: List[torch.Tensor]):
    if not root.requires_grad:
        return [None] * len(args)
    return torch.autograd.grad(root, args, allow_unused=True, retain_graph=True)


__all__ = [
    "get_dependencies",
]
