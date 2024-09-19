# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import numbers
import random
import sys
import timeit
import warnings
from collections import defaultdict
from contextlib import contextmanager
from itertools import zip_longest
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Union,
    overload,
)

import numpy as np
import torch

from pyro.poutine.util import site_is_subsample

if TYPE_CHECKING:
    from pyro.distributions.torch_distribution import TorchDistributionMixin
    from pyro.poutine.indep_messenger import CondIndepStackFrame
    from pyro.poutine.runtime import Message
    from pyro.poutine.trace import Trace


def set_rng_seed(rng_seed: int) -> None:
    """
    Sets seeds of `torch` and `torch.cuda` (if available).

    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)


def get_rng_state() -> Dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "random": random.getstate(),
        "numpy": np.random.get_state(),
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["random"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])


@overload
def torch_isnan(x: numbers.Number) -> bool: ...
@overload
def torch_isnan(x: torch.Tensor) -> torch.Tensor: ...
def torch_isnan(x: Union[torch.Tensor, numbers.Number]) -> Union[bool, torch.Tensor]:
    """
    A convenient function to check if a Tensor contains any nan; also works with numbers
    """
    if isinstance(x, numbers.Number):
        return x != x
    return torch.isnan(x).any()


@overload
def torch_isinf(x: numbers.Number) -> bool: ...
@overload
def torch_isinf(x: torch.Tensor) -> torch.Tensor: ...
def torch_isinf(x: Union[torch.Tensor, numbers.Number]) -> Union[bool, torch.Tensor]:
    """
    A convenient function to check if a Tensor contains any +inf; also works with numbers
    """
    if isinstance(x, numbers.Number):
        return x == math.inf or x == -math.inf
    return (x == math.inf).any() or (x == -math.inf).any()


@overload
def warn_if_nan(
    value: numbers.Number,
    msg: str = "",
    *,
    filename: Optional[str] = None,
    lineno: Optional[int] = None,
) -> numbers.Number: ...
@overload
def warn_if_nan(
    value: torch.Tensor,
    msg: str = "",
    *,
    filename: Optional[str] = None,
    lineno: Optional[int] = None,
) -> torch.Tensor: ...
def warn_if_nan(
    value: Union[torch.Tensor, numbers.Number],
    msg: str = "",
    *,
    filename: Optional[str] = None,
    lineno: Optional[int] = None,
) -> Union[torch.Tensor, numbers.Number]:
    """
    A convenient function to warn if a Tensor or its grad contains any nan,
    also works with numbers.
    """
    if filename is None:
        try:
            frame = sys._getframe(1)
        except ValueError:
            filename = "sys"
            lineno = 1
        else:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

    if isinstance(value, torch.Tensor) and value.requires_grad:
        value.register_hook(
            lambda x: warn_if_nan(
                x, "backward " + msg, filename=filename, lineno=lineno
            )
        )

    if torch_isnan(value):
        assert isinstance(lineno, int)
        warnings.warn_explicit(
            "Encountered NaN{}".format(": " + msg if msg else "."),
            UserWarning,
            filename,
            lineno,
        )

    return value


@overload
def warn_if_inf(
    value: numbers.Number,
    msg: str = "",
    allow_posinf: bool = False,
    allow_neginf: bool = False,
    *,
    filename: Optional[str] = None,
    lineno: Optional[int] = None,
) -> numbers.Number: ...
@overload
def warn_if_inf(
    value: torch.Tensor,
    msg: str = "",
    allow_posinf: bool = False,
    allow_neginf: bool = False,
    *,
    filename: Optional[str] = None,
    lineno: Optional[int] = None,
) -> torch.Tensor: ...
def warn_if_inf(
    value: Union[torch.Tensor, numbers.Number],
    msg: str = "",
    allow_posinf: bool = False,
    allow_neginf: bool = False,
    *,
    filename: Optional[str] = None,
    lineno: Optional[int] = None,
) -> Union[torch.Tensor, numbers.Number]:
    """
    A convenient function to warn if a Tensor or its grad contains any inf,
    also works with numbers.
    """
    if filename is None:
        try:
            frame = sys._getframe(1)
        except ValueError:
            filename = "sys"
            lineno = 1
        else:
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

    if isinstance(value, torch.Tensor) and value.requires_grad:
        value.register_hook(
            lambda x: warn_if_inf(
                x,
                "backward " + msg,
                allow_posinf,
                allow_neginf,
                filename=filename,
                lineno=lineno,
            )
        )

    if (not allow_posinf) and (
        value == math.inf
        if isinstance(value, numbers.Number)
        else (value == math.inf).any()
    ):
        assert isinstance(lineno, int)
        warnings.warn_explicit(
            "Encountered +inf{}".format(": " + msg if msg else "."),
            UserWarning,
            filename,
            lineno,
        )
    if (not allow_neginf) and (
        value == -math.inf
        if isinstance(value, numbers.Number)
        else (value == -math.inf).any()
    ):
        assert isinstance(lineno, int)
        warnings.warn_explicit(
            "Encountered -inf{}".format(": " + msg if msg else "."),
            UserWarning,
            filename,
            lineno,
        )

    return value


def save_visualization(trace: "Trace", graph_output: str) -> None:
    """
    DEPRECATED Use :func:`pyro.infer.inspect.render_model()` instead.

    Take a trace generated by poutine.trace with `graph_type='dense'`
    and render the graph with the output saved to file.

    - non-reparameterized stochastic nodes are salmon
    - reparameterized stochastic nodes are half salmon, half grey
    - observation nodes are green

    :param pyro.poutine.Trace trace: a trace to be visualized
    :param graph_output: the graph will be saved to graph_output.pdf
    :type graph_output: str

    Example::

        trace = pyro.poutine.trace(model, graph_type="dense").get_trace()
        save_visualization(trace, 'output')
    """
    warnings.warn(
        "`save_visualization` function is deprecated and will be removed in "
        "a future version."
    )

    import graphviz

    g = graphviz.Digraph()

    for label, node in trace.nodes.items():
        if site_is_subsample(node):
            continue
        shape = "ellipse"
        if label in trace.stochastic_nodes and label not in trace.reparameterized_nodes:
            fillcolor = "salmon"
        elif label in trace.reparameterized_nodes:
            fillcolor = "lightgrey;.5:salmon"
        elif label in trace.observation_nodes:
            fillcolor = "darkolivegreen3"
        else:
            # only visualize RVs
            continue
        g.node(label, label=label, shape=shape, style="filled", fillcolor=fillcolor)

    for label1, label2 in trace.edges:
        if site_is_subsample(trace.nodes[label1]):
            continue
        if site_is_subsample(trace.nodes[label2]):
            continue
        g.edge(label1, label2)

    g.render(graph_output, view=False, cleanup=True)


def check_traces_match(trace1: "Trace", trace2: "Trace") -> None:
    """
    :param pyro.poutine.Trace trace1: Trace object of the model
    :param pyro.poutine.Trace trace2: Trace object of the guide
    :raises: RuntimeWarning, ValueError

    Checks that (1) there is a bijection between the samples in the two traces
    and (2) at each sample site two traces agree on sample shape.
    """
    # Check ordinary sample sites.
    vars1 = set(name for name, site in trace1.nodes.items() if site["type"] == "sample")
    vars2 = set(name for name, site in trace2.nodes.items() if site["type"] == "sample")
    if vars1 != vars2:
        warnings.warn("Model vars changed: {} vs {}".format(vars1, vars2))

    # Check shapes agree.
    for name in vars1:
        site1 = trace1.nodes[name]
        site2 = trace2.nodes[name]
        if hasattr(site1["fn"], "shape") and hasattr(site2["fn"], "shape"):
            shape1 = site1["fn"].shape(*site1["args"], **site1["kwargs"])
            shape2 = site2["fn"].shape(*site2["args"], **site2["kwargs"])
            if shape1 != shape2:
                raise ValueError(
                    "Site dims disagree at site '{}': {} vs {}".format(
                        name, shape1, shape2
                    )
                )


def check_model_guide_match(
    model_trace: "Trace", guide_trace: "Trace", max_plate_nesting: float = math.inf
) -> None:
    """
    :param pyro.poutine.Trace model_trace: Trace object of the model
    :param pyro.poutine.Trace guide_trace: Trace object of the guide
    :raises: RuntimeWarning, ValueError

    Checks the following assumptions:
    1. Each sample site in the model also appears in the guide and is not
        marked auxiliary.
    2. Each sample site in the guide either appears in the model or is marked,
        auxiliary via ``infer={'is_auxiliary': True}``.
    3. Each :class:``~pyro.plate`` statement in the guide also appears in the
        model.
    4. At each sample site that appears in both the model and guide, the model
        and guide agree on sample shape.
    5. Model-side sequential enumeration is not implemented.
    """
    # Check ordinary sample sites.
    guide_vars = set(
        name
        for name, site in guide_trace.nodes.items()
        if site["type"] == "sample"
        if type(site["fn"]).__name__ != "_Subsample"
    )
    aux_vars = set(
        name
        for name, site in guide_trace.nodes.items()
        if site["type"] == "sample"
        if site["infer"].get("is_auxiliary")
    )
    model_vars = set(
        name
        for name, site in model_trace.nodes.items()
        if site["type"] == "sample" and not site["is_observed"]
        if type(site["fn"]).__name__ != "_Subsample"
    )
    enum_vars = set(
        name
        for name, site in model_trace.nodes.items()
        if site["type"] == "sample" and not site["is_observed"]
        if type(site["fn"]).__name__ != "_Subsample"
        if site["infer"].get("_enumerate_dim") is not None
        if name not in guide_vars
    )
    if aux_vars & model_vars:
        warnings.warn(
            "Found auxiliary vars in the model: {}".format(aux_vars & model_vars)
        )
    if not (guide_vars <= model_vars | aux_vars):
        warnings.warn(
            "Found non-auxiliary vars in guide but not model, "
            "consider marking these infer={{'is_auxiliary': True}}:\n{}".format(
                guide_vars - aux_vars - model_vars
            )
        )
    if not (model_vars <= guide_vars | enum_vars):
        bad_sites = model_vars - guide_vars - enum_vars
        for name in bad_sites:
            if model_trace.nodes[name]["infer"].get("enumerate") == "sequential":
                raise NotImplementedError(
                    f"At site {repr(name)}, "
                    "model-side sequential enumeration is not implemented. "
                    "Try parallel enumeration or guide-side enumeration."
                )
        warnings.warn(f"Found vars in model but not guide: {bad_sites}")

    # Check shapes agree.
    for name in model_vars & guide_vars:
        model_site = model_trace.nodes[name]
        guide_site = guide_trace.nodes[name]

        if hasattr(model_site["fn"], "event_dim") and hasattr(
            guide_site["fn"], "event_dim"
        ):
            if model_site["fn"].event_dim != guide_site["fn"].event_dim:
                raise ValueError(
                    "Model and guide event_dims disagree at site '{}': {} vs {}".format(
                        name, model_site["fn"].event_dim, guide_site["fn"].event_dim
                    )
                )

        if hasattr(model_site["fn"], "shape") and hasattr(guide_site["fn"], "shape"):
            model_shape = model_site["fn"].shape(
                *model_site["args"], **model_site["kwargs"]
            )
            guide_shape = guide_site["fn"].shape(
                *guide_site["args"], **guide_site["kwargs"]
            )
            if model_shape == guide_shape:
                continue

            # Allow broadcasting outside of max_plate_nesting.
            if len(model_shape) > max_plate_nesting:
                model_shape = model_shape[
                    len(model_shape) - max_plate_nesting - model_site["fn"].event_dim :
                ]
            if len(guide_shape) > max_plate_nesting:
                guide_shape = guide_shape[
                    len(guide_shape) - max_plate_nesting - guide_site["fn"].event_dim :
                ]
            if model_shape == guide_shape:
                continue
            for model_size, guide_size in zip_longest(
                reversed(model_shape), reversed(guide_shape), fillvalue=1
            ):
                if model_size != guide_size:
                    raise ValueError(
                        "Model and guide shapes disagree at site '{}': {} vs {}".format(
                            name, model_shape, guide_shape
                        )
                    )

    # Check subsample sites introduced by plate.
    model_vars = set(
        name
        for name, site in model_trace.nodes.items()
        if site["type"] == "sample" and not site["is_observed"]
        if type(site["fn"]).__name__ == "_Subsample"
    )
    guide_vars = set(
        name
        for name, site in guide_trace.nodes.items()
        if site["type"] == "sample"
        if type(site["fn"]).__name__ == "_Subsample"
    )
    if not (guide_vars <= model_vars):
        warnings.warn(
            "Found plate statements in guide but not model: {}".format(
                guide_vars - model_vars
            )
        )

    # Check factor statements in guide specify has_rsample.
    for name, site in guide_trace.nodes.items():
        if not site["type"] == "sample":
            continue
        if not site["infer"].get("is_auxiliary"):
            continue
        if type(site["fn"]).__name__ != "Unit":
            continue
        if "has_rsample" not in site["fn"].__dict__:
            raise ValueError(
                f'At guide site pyro.factor("{name}",...), '
                "missing specification of has_rsample. "
                "Please either set has_rsample=True if the factor statement arises "
                "from reparametrized sampling or has_rsample=False otherwise."
            )


def check_site_shape(site: "Message", max_plate_nesting: int) -> None:
    actual_shape = list(site["log_prob"].shape)

    # Compute expected shape.
    expected_shape: List[Optional[int]] = []
    for f in site["cond_indep_stack"]:
        if f.dim is not None:
            # Use the specified plate dimension, which counts from the right.
            assert f.dim < 0
            if len(expected_shape) < -f.dim:
                extra_shape: List[Optional[int]] = [None] * (
                    -f.dim - len(expected_shape)
                )
                expected_shape = extra_shape + expected_shape
            if expected_shape[f.dim] is not None:
                raise ValueError(
                    "\n  ".join(
                        [
                            'at site "{}" within plate("{}", dim={}), dim collision'.format(
                                site["name"], f.name, f.dim
                            ),
                            "Try setting dim arg in other plates.",
                        ]
                    )
                )
            expected_shape[f.dim] = f.size
    expected_shape = [-1 if e is None else e for e in expected_shape]

    # Check for plate stack overflow.
    if len(expected_shape) > max_plate_nesting:
        raise ValueError(
            "\n  ".join(
                [
                    'at site "{}", plate stack overflow'.format(site["name"]),
                    "Try increasing max_plate_nesting to at least {}".format(
                        len(expected_shape)
                    ),
                ]
            )
        )

    # Ignore dimensions left of max_plate_nesting.
    if max_plate_nesting < len(actual_shape):
        actual_shape = actual_shape[len(actual_shape) - max_plate_nesting :]

    # Check for incorrect plate placement on the right of max_plate_nesting.
    for actual_size, expected_size in zip_longest(
        reversed(actual_shape), reversed(expected_shape), fillvalue=1
    ):
        if expected_size != -1 and expected_size != actual_size:
            raise ValueError(
                "\n  ".join(
                    [
                        'at site "{}", invalid log_prob shape'.format(site["name"]),
                        "Expected {}, actual {}".format(expected_shape, actual_shape),
                        "Try one of the following fixes:",
                        "- enclose the batched tensor in a with pyro.plate(...): context",
                        "- .to_event(...) the distribution being sampled",
                        "- .permute() data dimensions",
                    ]
                )
            )

    # Check parallel dimensions on the left of max_plate_nesting.
    if TYPE_CHECKING:
        assert site["infer"] is not None
        assert isinstance(site["fn"], TorchDistributionMixin)
    enum_dim = site["infer"].get("_enumerate_dim")
    if enum_dim is not None:
        if (
            len(site["fn"].batch_shape) >= -enum_dim
            and site["fn"].batch_shape[enum_dim] != 1
        ):
            raise ValueError(
                "\n  ".join(
                    [
                        'Enumeration dim conflict at site "{}"'.format(site["name"]),
                        "Try increasing pyro.markov history size",
                    ]
                )
            )


def _are_independent(counters1: Dict[str, int], counters2: Dict[str, int]) -> bool:
    for name, counter1 in counters1.items():
        if name in counters2:
            if counters2[name] != counter1:
                return True
    return False


def check_traceenum_requirements(model_trace: "Trace", guide_trace: "Trace") -> None:
    """
    Warn if user could easily rewrite the model or guide in a way that would
    clearly avoid invalid dependencies on enumerated variables.

    :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO` enumerates over
    synchronized products rather than full cartesian products. Therefore models
    must ensure that no variable outside of an plate depends on an enumerated
    variable inside that plate. Since full dependency checking is impossible,
    this function aims to warn only in cases where models can be easily
    rewitten to be obviously correct.
    """
    enumerated_sites = set(
        name
        for name, site in guide_trace.nodes.items()
        if site["type"] == "sample" and site["infer"].get("enumerate")
    )
    for role, trace in [("model", model_trace), ("guide", guide_trace)]:
        plate_counters: Dict[str, Dict[str, int]] = {}  # for sequential plates only
        enumerated_contexts: Dict[FrozenSet["CondIndepStackFrame"], Set[str]] = (
            defaultdict(set)
        )
        for name, site in trace.nodes.items():
            if site["type"] != "sample":
                continue
            plate_counter = {
                f.name: f.counter for f in site["cond_indep_stack"] if not f.vectorized
            }
            context = frozenset(f for f in site["cond_indep_stack"] if f.vectorized)

            # Check that sites outside each independence context precede enumerated sites inside that context.
            for enumerated_context, names in enumerated_contexts.items():
                if not (context < enumerated_context):
                    continue
                names_list = sorted(
                    n
                    for n in names
                    if not _are_independent(plate_counter, plate_counters[n])
                )
                if not names_list:
                    continue
                diff = sorted(f.name for f in enumerated_context - context)
                warnings.warn(
                    "\n  ".join(
                        [
                            'at {} site "{}", possibly invalid dependency.'.format(
                                role, name
                            ),
                            'Expected site "{}" to precede sites "{}"'.format(
                                name, '", "'.join(sorted(names_list))
                            ),
                            'to avoid breaking independence of plates "{}"'.format(
                                '", "'.join(diff)
                            ),
                        ]
                    ),
                    RuntimeWarning,
                )

            plate_counters[name] = plate_counter
            if name in enumerated_sites:
                enumerated_contexts[context].add(name)


def check_if_enumerated(guide_trace: "Trace") -> None:
    enumerated_sites = [
        name
        for name, site in guide_trace.nodes.items()
        if site["type"] == "sample" and site["infer"].get("enumerate")
    ]
    if enumerated_sites:
        warnings.warn(
            "\n".join(
                [
                    "Found sample sites configured for enumeration:"
                    ", ".join(enumerated_sites),
                    "If you want to enumerate sites, you need to use TraceEnum_ELBO instead.",
                ]
            )
        )


@contextmanager
def ignore_jit_warnings(filter=None):
    """
    Ignore JIT tracer warnings with messages that match `filter`. If
    `filter` is not specified all tracer warnings are ignored.

    Note this only installs warning filters if executed within traced code.

    :param filter: A list containing either warning message (str),
        or tuple consisting of (warning message (str), Warning class).
    """
    if not torch._C._get_tracing_state():
        yield
        return

    with warnings.catch_warnings():
        if filter is None:
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        else:
            for msg in filter:
                category = torch.jit.TracerWarning
                if isinstance(msg, tuple):
                    msg, category = msg
                warnings.filterwarnings("ignore", category=category, message=msg)
        yield


def jit_iter(tensor: torch.Tensor) -> List[torch.Tensor]:
    """
    Iterate over a tensor, ignoring jit warnings.
    """
    # The "Iterating over a tensor" warning is erroneously a RuntimeWarning
    # so we use a custom filter here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Iterating over a tensor")
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        return list(tensor)


class optional:
    """
    Optionally wrap inside `context_manager` if condition is `True`.
    """

    def __init__(self, context_manager, condition):
        self.context_manager = context_manager
        self.condition = condition

    def __enter__(self):
        if self.condition:
            return self.context_manager.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.condition:
            return self.context_manager.__exit__(exc_type, exc_val, exc_tb)


class ExperimentalWarning(UserWarning):
    pass


@contextmanager
def ignore_experimental_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        yield


class timed:
    def __enter__(self, timer=timeit.default_timer):
        self.start = timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = timeit.default_timer()
        self.elapsed = self.end - self.start
        return self.elapsed


@overload
def torch_float(x: Union[float, int]) -> float: ...
@overload
def torch_float(x: torch.Tensor) -> torch.Tensor: ...
def torch_float(
    x: Union[torch.Tensor, Union[float, int]]
) -> Union[torch.Tensor, float]:
    return x.float() if isinstance(x, torch.Tensor) else float(x)
