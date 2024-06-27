# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Collection, Dict, List, Optional, Union

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.ops.provenance import ProvenanceTensor, detach_provenance, get_provenance
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample

try:
    import graphviz
except ImportError:
    graphviz = SimpleNamespace(Digraph=object)  # for type hints


def is_sample_site(msg, *, include_deterministic=False):
    if msg["type"] != "sample":
        return False
    if site_is_subsample(msg):
        return False

    if not include_deterministic:
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


def site_is_deterministic(msg: dict) -> bool:
    return msg["type"] == "sample" and msg["infer"].get("_deterministic", False)


class TrackProvenance(Messenger):
    def __init__(self, *, include_deterministic=False):
        self.include_deterministic = include_deterministic

    def _pyro_post_sample(self, msg):
        if self.include_deterministic and site_is_deterministic(msg):
            provenance = frozenset({msg["name"]})  # track only direct dependencies
            value = detach_provenance(msg["value"])
            msg["value"] = ProvenanceTensor(value, provenance)

        elif is_sample_site(msg, include_deterministic=self.include_deterministic):
            provenance = frozenset({msg["name"]})  # track only direct dependencies
            value = detach_provenance(msg["value"])
            msg["value"] = ProvenanceTensor(value, provenance)

    def _pyro_post_param(self, msg):
        if msg["type"] == "param":
            provenance = frozenset({msg["name"]})  # track only direct dependencies
            value = detach_provenance(msg["value"])
            msg["value"] = ProvenanceTensor(value, provenance)


@torch.enable_grad()
def get_dependencies(
    model: Callable,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
    include_deterministic: bool = False,
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
        variables to the latent or observed sites on which they depend in the
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
    in the posterior to the so-called collider variable ``c`` on which they
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
    we see that ``a`` depends on itself across the plate ``p``. This means
    that, among the elements of ``a``, e.g. ``a[0]`` depends on ``a[1]`` (this
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

    [1] S.Webb, A.Goliński, R.Zinkov, N.Siddharth, T.Rainforth, Y.W.Teh, F.Wood (2018)
        "Faithful inversion of generative models for effective amortized inference"
        https://dl.acm.org/doi/10.5555/3327144.3327229

    :param callable model: A model.
    :param tuple model_args: Optional tuple of model args.
    :param dict model_kwargs: Optional dict of model kwargs.
    :param bool include_deterministic: Whether to include deterministic sites.
    :returns: A dictionary of metadata (see above).
    :rtype: dict
    """
    if model_args is None:
        model_args = ()
    if model_kwargs is None:
        model_kwargs = {}

    # Collect sites with tracked provenance.
    with torch.random.fork_rng(), torch.no_grad(), pyro.validation_enabled(False):
        with TrackProvenance(include_deterministic=include_deterministic):
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


def get_model_relations(
    model: Callable,
    model_args: Optional[tuple] = None,
    model_kwargs: Optional[dict] = None,
    include_deterministic: bool = False,
):
    """
    Infer relations of RVs and plates from given model and optionally data.
    See https://github.com/pyro-ppl/pyro/issues/949 for more details.

    This returns a dictionary with keys:

    -  "sample_sample" map each downstream sample site to a list of the upstream
       sample sites on which it depend;
    -  "sample_dist" maps each sample site to the name of the distribution at
       that site;
    -  "plate_sample" maps each plate name to a list of the sample sites within
       that plate; and
    -  "observe" is a list of observed sample sites.

    For example for the model::

        def model(data):
            m = pyro.sample('m', dist.Normal(0, 1))
            sd = pyro.sample('sd', dist.LogNormal(m, 1))
            with pyro.plate('N', len(data)):
                pyro.sample('obs', dist.Normal(m, sd), obs=data)

    the relation is::

        {'sample_sample': {'m': [], 'sd': ['m'], 'obs': ['m', 'sd']},
         'sample_dist': {'m': 'Normal', 'sd': 'LogNormal', 'obs': 'Normal'},
         'plate_sample': {'N': ['obs']},
         'observed': ['obs']}

    :param callable model: A model to inspect.
    :param model_args: Optional tuple of model args.
    :param model_kwargs: Optional dict of model kwargs.
    :param bool include_deterministic: Whether to include deterministic sites.
    :rtype: dict
    """
    if model_args is None:
        model_args = ()
    if model_kwargs is None:
        model_kwargs = {}
    assert isinstance(model_args, tuple)
    assert isinstance(model_kwargs, dict)

    with torch.random.fork_rng(), torch.no_grad(), pyro.validation_enabled(False):
        with TrackProvenance(include_deterministic=include_deterministic):
            trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)

    sample_sample = {}
    sample_param = {}
    sample_dist = {}
    param_constraint = {}
    plate_sample = defaultdict(list)
    observed = []

    def _get_type_from_frozenname(frozen_name):
        return trace.nodes[frozen_name]["type"]

    for name, site in trace.nodes.items():
        if site["type"] == "param":
            param_constraint[name] = str(site["kwargs"]["constraint"])

        if site["type"] != "sample" or site_is_subsample(site):
            continue

        provenance = get_provenance(
            site["fn"].log_prob(site["value"])
            if not site_is_deterministic(site)
            else site["fn"].base_dist.log_prob(site["value"])
        )
        sample_sample[name] = [
            upstream
            for upstream in provenance
            if upstream != name and _get_type_from_frozenname(upstream) == "sample"
        ]

        sample_param[name] = [
            upstream
            for upstream in provenance
            if upstream != name and _get_type_from_frozenname(upstream) == "param"
        ]

        sample_dist[name] = (
            _get_dist_name(site["fn"])
            if not site_is_deterministic(site)
            else "Deterministic"
        )
        for frame in site["cond_indep_stack"]:
            plate_sample[frame.name].append(name)
        if site["is_observed"]:
            observed.append(name)

    def _resolve_plate_samples(plate_samples):
        for p, pv in plate_samples.items():
            pv = set(pv)
            for q, qv in plate_samples.items():
                qv = set(qv)
                if len(pv & qv) > 0 and len(pv - qv) > 0 and len(qv - pv) > 0:
                    plate_samples_ = plate_samples.copy()
                    plate_samples_[q] = pv & qv
                    plate_samples_[q + "__CLONE"] = qv - pv
                    return _resolve_plate_samples(plate_samples_)
        return plate_samples

    plate_sample = _resolve_plate_samples(plate_sample)

    # Normalize order of variables.
    def sort_by_time(names: Collection[str]) -> List[str]:
        return [name for name in trace.nodes if name in names]

    sample_sample = {k: sort_by_time(v) for k, v in sample_sample.items()}
    sample_param = {k: sort_by_time(v) for k, v in sample_param.items()}
    plate_sample = {k: sort_by_time(v) for k, v in plate_sample.items()}
    observed = sort_by_time(observed)

    return {
        "sample_sample": sample_sample,
        "sample_param": sample_param,
        "sample_dist": sample_dist,
        "param_constraint": param_constraint,
        "plate_sample": dict(plate_sample),
        "observed": observed,
    }


def _get_dist_name(fn):
    while isinstance(
        fn, (dist.Independent, dist.ExpandedDistribution, dist.MaskedDistribution)
    ):
        fn = fn.base_dist
    return type(fn).__name__


def generate_graph_specification(
    model_relations: dict, render_params: bool = False
) -> dict:
    """
    Convert model relations into data structure which can be readily
    converted into a network.
    """
    # group nodes by plate
    plate_groups = dict(model_relations["plate_sample"])
    plate_rvs = {rv for rvs in plate_groups.values() for rv in rvs}
    plate_groups[None] = [
        rv for rv in model_relations["sample_sample"] if rv not in plate_rvs
    ]  # RVs which are in no plate

    # get set of params
    params = set()
    if render_params:
        for rv, params_list in model_relations["sample_param"].items():
            for param in params_list:
                params.add(param)
        plate_groups[None].extend(params)

    # retain node metadata
    node_data = {}
    for rv in model_relations["sample_sample"]:
        node_data[rv] = {
            "is_observed": rv in model_relations["observed"],
            "distribution": model_relations["sample_dist"][rv],
        }

    if render_params:
        for param, constraint in model_relations["param_constraint"].items():
            node_data[param] = {
                "is_observed": False,
                "constraint": constraint,
                "distribution": None,
            }

    # infer plate structure
    # (when the order of plates cannot be determined from subset relations,
    # it follows the order in which plates appear in trace)
    plate_data = {}
    for plate1, plate2 in list(itertools.combinations(plate_groups, 2)):
        if plate1 is None or plate2 is None:
            continue

        nodes1 = set(plate_groups[plate1])
        nodes2 = set(plate_groups[plate2])
        if nodes1 < nodes2:
            plate_data[plate1] = {"parent": plate2}
        elif nodes1 >= nodes2:
            plate_data[plate2] = {"parent": plate1}
        elif nodes1 & nodes2:
            raise NotImplementedError(
                f"Overlapping non-nested plates {repr(plate1)},{repr(plate2)} "
                "are not supported by render_model(). To help add support see "
                "https://github.com/pyro-ppl/pyro/issues/2980"
            )

    for plate in plate_groups:
        if plate is None:
            continue

        if plate not in plate_data:
            plate_data[plate] = {"parent": None}

    # infer RV edges
    edge_list = []
    for target, source_list in model_relations["sample_sample"].items():
        edge_list.extend([(source, target) for source in source_list])

    if render_params:
        for target, source_list in model_relations["sample_param"].items():
            edge_list.extend([(source, target) for source in source_list])

    return {
        "plate_groups": plate_groups,
        "plate_data": plate_data,
        "node_data": node_data,
        "edge_list": edge_list,
    }


def _deep_merge(things: list):
    if len(things) == 1:
        return things[0]

    # Recurse into dicts.
    if isinstance(things[0], dict):
        result = {}
        for thing in things:
            for key, value in thing.items():
                if key not in result:
                    result[key] = _deep_merge([t[key] for t in things])
        return result

    # Vote for booleans.
    if isinstance(things[0], bool):
        if all(x is True for x in things):
            return True
        if all(x is False for x in things):
            return False
        return None  # i.e. maybe

    # Otherwise choose arbitrarily.
    return things[0]


def render_graph(
    graph_specification: dict, render_distributions: bool = False
) -> "graphviz.Digraph":
    """
    Create a graphviz object given a graph specification.

    :param bool render_distributions: Show distribution of each RV in plot.
    """
    try:
        import graphviz  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Looks like you want to use graphviz (https://graphviz.org/) "
            "to render your model. "
            "You need to install `graphviz` to be able to use this feature. "
            "It can be installed with `pip install graphviz`."
        ) from e

    plate_groups = graph_specification["plate_groups"]
    plate_data = graph_specification["plate_data"]
    node_data = graph_specification["node_data"]
    edge_list = graph_specification["edge_list"]

    graph = graphviz.Digraph()

    # add plates
    plate_graph_dict = {
        plate: graphviz.Digraph(name=f"cluster_{plate}")
        for plate in plate_groups
        if plate is not None
    }
    for plate, plate_graph in plate_graph_dict.items():
        plate_graph.attr(label=plate.split("__CLONE")[0], labeljust="r", labelloc="b")

    plate_graph_dict[None] = graph

    # add nodes
    colors = {False: "white", True: "gray", None: "gray:white"}
    for plate, rv_list in plate_groups.items():
        cur_graph = plate_graph_dict[plate]

        for rv in rv_list:
            color = colors[node_data[rv]["is_observed"]]

            # For sample_nodes - ellipse
            if node_data[rv]["distribution"]:
                shape = "ellipse"
                rv_label = rv

            # For param_nodes - No shape
            else:
                shape = "plain"
                rv_label = rv.replace("$params", "")

            # use different symbol for Deterministic site
            node_style = (
                "filled,dashed"
                if node_data[rv]["distribution"] == "Deterministic"
                else "filled"
            )
            cur_graph.node(
                rv, label=rv_label, shape=shape, style=node_style, fillcolor=color
            )

    # add leaf nodes first
    while len(plate_data) >= 1:
        for plate, data in plate_data.items():
            parent_plate = data["parent"]
            is_leaf = True

            for plate2, data2 in plate_data.items():
                if plate == data2["parent"]:
                    is_leaf = False
                    break

            if is_leaf:
                plate_graph_dict[parent_plate].subgraph(plate_graph_dict[plate])
                plate_data.pop(plate)
                break

    # add edges
    for source, target in edge_list:
        graph.edge(source, target)

    # render distributions if requested
    if render_distributions:
        dist_label = ""
        for rv, data in node_data.items():
            rv_dist = data["distribution"]
            if rv_dist:
                dist_label += rf"{rv} ~ {rv_dist}\l"

            if "constraint" in data and data["constraint"]:
                dist_label += rf"{rv} : {data['constraint']}\l"

        graph.node("distribution_description_node", label=dist_label, shape="plaintext")

    # return whole graph
    return graph


def render_model(
    model: Callable,
    model_args: Optional[Union[tuple, List[tuple]]] = None,
    model_kwargs: Optional[Union[dict, List[dict]]] = None,
    filename: Optional[str] = None,
    render_distributions: bool = False,
    render_params: bool = False,
    render_deterministic: bool = False,
) -> "graphviz.Digraph":
    """
    Renders a model using `graphviz <https://graphviz.org>`_ .

    If ``filename`` is provided, this saves an image; otherwise this draws the
    graph. For example usage see the
    `model rendering tutorial <https://pyro.ai/examples/model_rendering.html>`_ .

    :param model: Model to render.
    :param model_args: Tuple of positional arguments to pass to the model, or
        list of tuples for semisupervised models.
    :param model_kwargs: Dict of keyword arguments to pass to the model, or
        list of dicts for semisupervised models.
    :param str filename: Name of file or path to file to save rendered model in.
    :param bool render_distributions: Whether to include RV distribution
        annotations (and param constraints) in the plot.
    :param bool render_params: Whether to show params inthe plot.
    :param bool render_deterministic: Whether to include deterministic sites.
    :returns: A model graph.
    :rtype: graphviz.Digraph
    """
    # Get model relations.
    if not isinstance(model_args, list) and not isinstance(model_kwargs, list):
        relations = [
            get_model_relations(
                model,
                model_args,
                model_kwargs,
                include_deterministic=render_deterministic,
            )
        ]
    else:  # semisupervised
        if isinstance(model_args, list):
            if not isinstance(model_kwargs, list):
                model_kwargs = [model_kwargs] * len(model_args)
        elif not isinstance(model_args, list):
            model_args = [model_args] * len(model_kwargs)
        assert len(model_args) == len(model_kwargs)
        relations = [
            get_model_relations(
                model, args, kwargs, include_deterministic=render_deterministic
            )
            for args, kwargs in zip(model_args, model_kwargs)
        ]

    # Get graph specifications.
    graph_specs = [
        generate_graph_specification(r, render_params=render_params) for r in relations
    ]
    graph_spec = _deep_merge(graph_specs)

    # Render.
    graph = render_graph(graph_spec, render_distributions=render_distributions)

    if filename is not None:
        suffix = Path(filename).suffix[1:]  # remove leading period from suffix
        filepath = os.path.splitext(filename)[0]
        graph.render(filepath, view=False, cleanup=True, format=suffix)

    return graph


__all__ = [
    "get_dependencies",
    "render_model",
]
