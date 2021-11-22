# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, NamedTuple, Set, TypeVar, Union

import torch
from torch import Tensor, tensor

from pyro.poutine import Trace
from pyro.poutine.handlers import _make_handler
from pyro.poutine.messenger import Messenger
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.trace_messenger import TraceMessenger

# type aliases

Node = dict
T = TypeVar("T")
Predicate = Callable[[Any], bool]
SiteFilter = Callable[[str, Node], bool]


def concat_traces(
    *traces: Trace,
    site_filter: Callable[[str, Any], bool] = lambda name, node: True,
) -> Trace:
    newtrace = Trace()
    for tr in traces:
        drop_edges = []
        for name, node in tr.nodes.items():
            if site_filter(name, node):
                newtrace.add_node(name, **node)
            else:
                drop_edges.append(name)
        for p, s in zip(tr._pred, tr._succ):
            if p not in drop_edges and s not in drop_edges:
                newtrace.add_edge(p, s)
    return newtrace


def _addrs(tr):
    return {k for k, n in tr.nodes.items() if n["type"] == "sample"}


def assert_no_overlap(t0: Trace, t1: Trace, location=""):
    assert (
        len(_addrs(t0).intersection(_addrs(t1))) == 0
    ), f"{location}: addresses must not overlap"


def is_observed(node: Node) -> bool:
    return node.get("is_observed", False)


def is_sample_type(node: Node) -> bool:
    return node["type"] == "sample"


def not_observed(node: Node) -> bool:
    return not is_observed(node)


def _check_infer_map(k: str) -> Callable[[Node], bool]:
    return lambda node: node.get("infer", dict()).get(k, False)


is_substituted = _check_infer_map("substituted")


def not_substituted(node: Node) -> bool:
    return not is_substituted(node)


is_auxiliary = _check_infer_map("is_auxiliary")


def not_auxiliary(node: Node) -> bool:
    return not is_auxiliary(node)


def _or(p0: Predicate, p1: Predicate) -> Predicate:
    return lambda x: p0(x) or p1(x)


def node_filter(p: Callable[[Node], bool]) -> SiteFilter:
    return lambda _, node: p(node)


def sample_filter(p: Callable[[Node], bool]) -> SiteFilter:
    return lambda _, node: is_sample_type(node) and p(node)


def addr_filter(p: Callable[[str], bool]) -> SiteFilter:
    return lambda name, _: p(name)


def membership_filter(members: Set[str]) -> SiteFilter:
    return lambda name, _: name in members


# class Trace(NamedTuple):
#     output: Any
#     log_weight: Tensor
#     trace: Trace


class WithSubstitutionMessenger(ReplayMessenger):
    def _pyro_sample(self, msg):
        orig_infer = msg["infer"]
        super()._pyro_sample(msg)
        if (
            self.trace is not None
            and msg["name"] in self.trace
            and not msg["is_observed"]
        ):
            new_infer = msg["infer"]
            orig_infer.update(new_infer)
            orig_infer["substituted"] = True
            msg["infer"] = orig_infer


class AuxiliaryMessenger(Messenger):
    def __init__(self) -> None:
        super().__init__()

    def _pyro_sample(self, msg):
        msg["infer"]["is_auxiliary"] = True


for messenger in [WithSubstitutionMessenger, AuxiliaryMessenger]:
    _handler_name, _handler = _make_handler(messenger)
    _handler.__module__ = __name__
    locals()[_handler_name] = _handler


def get_marginal(trace: Trace) -> Trace:
    return concat_traces(trace, site_filter=sample_filter(not_auxiliary))


class inference(object):
    def __init__(self):
        self.loss = 0.0


Inference = Union[inference, Callable[..., Trace]]


class targets(inference):
    def __init__(self):
        super().__init__()


Targets = Union[targets, Callable[..., Trace]]


class proposals(inference):
    def __init__(self):
        super().__init__()


Proposals = Union[proposals, Callable[..., Trace]]


def stacked_log_prob(
    trace, site_filter: Callable[[str, Any], bool] = (lambda name, node: True)
):
    trace.compute_log_prob(site_filter=site_filter)
    log_probs = [
        n["log_prob"]
        for k, n in trace.nodes.items()
        if "log_prob" in n and site_filter(k, n)
    ]

    log_prob = (
        torch.stack(log_probs).sum(dim=0) if len(log_probs) > 0 else tensor([0.0])
    )
    if len(log_prob.shape) == 0:
        log_prob.unsqueeze_(0)
    return log_prob


class primitive(targets, proposals):
    def __init__(self, program: Callable[..., Any]):
        super().__init__()
        self.program = program

    def __call__(self, *args, **kwargs) -> Trace:
        with TraceMessenger() as tracer:
            out = self.program(*args, **kwargs)
            tr: Trace = tracer.trace
            lp = stacked_log_prob(tr, sample_filter(_or(is_substituted, is_observed)))

            trace = tr
            set_input(
                trace,
                args=args,
                kwargs=kwargs,
            )
            set_param(trace, _RETURN, "return", value=out)
            set_param(trace, _LOGWEIGHT, "return", value=lp)
            return trace


Primitive = Union[Callable[..., Trace], primitive]

_INPUT, _RETURN, _LOGWEIGHT, _LOSS = "_INPUT", "_RETURN", "_LOGWEIGHT", "_LOSS"


def set_input(trace, args, kwargs):
    trace.add_node("_INPUT", name="_INPUT", type="input", args=args, kwargs=kwargs)


def set_param(trace, name, type, value=None):
    trace.add_node(name, name=name, type=type, value=value)


def valueat(o, a):
    return o.nodes[a]["value"]


def _logweight(tr):
    return valueat(tr, "_LOGWEIGHT")


def _output(tr):
    return valueat(tr, "_RETURN")


def _addrs(tr):
    return {k for k, n in tr.nodes.items() if n["type"] == "sample"}


class extend(targets):
    def __init__(self, p: Targets, f: Primitive):
        super().__init__()
        self.p, self.f = p, f

    def __call__(self, *args, **kwargs) -> Trace:
        p_out: Trace = self.p(*args, **kwargs)
        f_out: Trace = auxiliary(self.f)(_output(p_out), *args, **kwargs)
        p_trace, f_trace = p_out, f_out

        # NOTE: we need to walk across all nodes to see if substitution was used
        under_substitution = any(
            map(is_substituted, [*p_trace.nodes.values(), *f_trace.nodes.values()])
        )

        assert_no_overlap(p_trace, f_trace, location=type(self))
        assert all(map(not_observed, f_trace.nodes.values()))
        if not under_substitution:
            assert _logweight(f_out) == 0.0
            assert all(map(not_substituted, f_trace.nodes.values()))

        log_u2 = stacked_log_prob(f_trace, site_filter=node_filter(is_sample_type))

        trace = concat_traces(p_out, f_out, site_filter=node_filter(is_sample_type))
        set_input(trace, args=args, kwargs=kwargs)
        set_param(trace, _RETURN, "return", value=_output(f_out))
        set_param(trace, _LOGWEIGHT, "return", value=_logweight(p_out) + log_u2)
        return trace


class compose(proposals):
    def __init__(self, q2: Proposals, q1: Proposals):
        super().__init__()
        self.q1, self.q2 = q1, q2

    def __call__(self, *args, **kwargs) -> Trace:
        q1_out = self.q1(*args, **kwargs)
        q2_out = self.q2(*args, **kwargs)

        assert_no_overlap(q2_out, q1_out, location=type(self))

        trace = concat_traces(q2_out, q1_out, site_filter=node_filter(is_sample_type))
        set_input(
            trace,
            args=args,
            kwargs=kwargs,
        )
        set_param(trace, _RETURN, "return", value=_output(q2_out))
        set_param(
            trace, _LOGWEIGHT, "return", value=_logweight(q1_out) + _logweight(q2_out)
        )
        return trace


class propose(proposals):
    def __init__(
        self,
        p: Targets,
        q: Proposals,
        loss_fn: Callable[[Trace], Union[Tensor, float]] = (lambda out: 0.0),
    ):
        super().__init__()
        self.p, self.q = p, q
        self.loss_fn = loss_fn

    def __call__(self, *args, **kwargs) -> Trace:
        q_out = self.q(*args, **kwargs)
        p_out = with_substitution(self.p, trace=q_out)(*args, **kwargs)

        # FIXME this hook exists to alter an NVI trace for stl
        # q_trace = dispatch(self.transf_q_trace, q_out.trace, **inf_kwargs)

        lu_1 = stacked_log_prob(
            q_out, site_filter=sample_filter(_or(not_substituted, is_observed))
        )
        lw_1 = _logweight(q_out)

        # We call that lv because its the incremental weight in the IS sense
        lv = _logweight(p_out) - lu_1
        lw_out = lw_1 + lv

        m_trace = get_marginal(p_out)
        # FIXME: this is not accounted for -- will return the final kernel output, not the initial output
        # should be something like: m_output = m_trace["_RETURN"]
        m_output = _output(p_out)

        # FIXME local gradient computations
        # trace=m_trace if self._no_reruns else rerun_with_detached_values(m_trace),
        trace = m_trace
        set_input(
            trace,
            args=args,
            kwargs=kwargs,
        )
        set_param(trace, _RETURN, "return", value=m_output)
        set_param(trace, _LOGWEIGHT, "return", value=lw_out.detach())
        self.loss = self.loss + self.loss_fn(m_output)
        return trace


class observe(proposals):
    def __init__(
        self,
        q: Proposals,
        loss_fn: Callable[[Trace], Union[Tensor, float]],
    ):
        super().__init__()
        self.q = q
        self.loss_fn = loss_fn
        self.loss = 0.0

    def __call__(self, *args, **kwargs) -> Trace:
        q_out = self.q(*args, **kwargs)
        loss = self.loss + self.loss_fn(m_output)

        return Trace(
            trace=concat_traces(
                q2_out.trace, q1_out.trace, site_filter=node_filter(is_sample_type)
            ),
            log_weight=q1_out.log_weight + q2_out.log_weight,
            output=q2_out.output,
        )
