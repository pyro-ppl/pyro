# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional

import torch

import pyro
import pyro.poutine as poutine
from pyro.ops.provenance import ProvenanceTensor, get_provenance
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


class TrackProvenance(Messenger):
    def _pyro_post_sample(self, msg):
        if is_sample_site(msg):
            provenance = frozenset({msg["name"]})  # track only direct dependencies
            value = msg["value"]
            while isinstance(value, ProvenanceTensor):
                value = value._t
            msg["value"] = ProvenanceTensor(value, provenance)


@torch.enable_grad()
def get_dependencies(
    model: Callable,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
) -> Dict[str, object]:
    r"""
    Infers dependency structure about a conditioned model.

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

    # Collect sites with tracked provenance.
    with torch.random.fork_rng(), torch.no_grad(), pyro.validation_enabled(False):
        with TrackProvenance():
            trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
    sample_sites = [msg for msg in trace.nodes.values() if is_sample_site(msg)]

    # Collect observations.
    observed = {msg["name"] for msg in sample_sites if msg["is_observed"]}
    plates = {
        msg["name"]: {f.name for f in msg["cond_indep_stack"] if f.vectorized}
        for msg in sample_sites
    }

    # Find direct prior dependencies among latent and observed sites.
    prior_dependencies = {n: {n: set()} for n in plates}  # no deps yet
    for i, downstream in enumerate(sample_sites):
        upstreams = [
            u for u in sample_sites[:i] if not u["is_observed"] if u["value"].numel()
        ]
        if not upstreams:
            continue
        log_prob = downstream["fn"].log_prob(downstream["value"])
        provenance = get_provenance(log_prob)
        for upstream in upstreams:
            u = upstream["name"]
            if u in provenance:
                d = downstream["name"]
                prior_dependencies[d][u] = set()

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


__all__ = [
    "get_dependencies",
]
