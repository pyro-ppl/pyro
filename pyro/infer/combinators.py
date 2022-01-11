# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import logging
import torch
from torch import Tensor, tensor
from typing import Any, Callable, Tuple, Set, TypeVar, Union

from pyro.poutine import Trace
from pyro.poutine.handlers import _make_handler
from pyro.poutine.messenger import Messenger
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.trace_messenger import TraceMessenger

logger = logging.getLogger(__name__)

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


def no_samples_overlap(t0: Trace, t1: Trace):
    return len(_addrs(t0).intersection(_addrs(t1))) == 0


def is_observed(node: Node) -> bool:
    return node.get("is_observed", False)


def is_sample_type(node: Node) -> bool:
    return node["type"] == "sample"


def is_return_type(node: Node) -> bool:
    return node["type"] == "return"


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


def get_marginal(trace: Trace) -> Tuple[Trace, Node]:
    m_output = trace.nodes[_RETURN]
    while "infer" in m_output:
        m_output = m_output["infer"]["m_return_node"]
    m_trace = concat_traces(trace, site_filter=sample_filter(not_auxiliary))
    return m_trace, m_output


class inference(object):
    # FIXME: needs something like an "infer" map to hold things like index for APG.
    # this must be placed at the top level on a trace in APG
    pass


Inference = Union[inference, Callable[..., Trace]]


class targets(inference):
    pass


Targets = Union[targets, Callable[..., Trace]]


class proposals(inference):
    pass


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
            tr: Trace = tracer.trace  # type: ignore
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

    def __repr__(self) -> str:
        return f"<{type(self).__name__} wrapper of {repr(self.program)} at {hex(id(self))}>"


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
    return {k for k, n in tr.nodes.items() if is_sample_type(n)}


class extend(targets):
    def __init__(self, p: Targets, f: Primitive):
        super().__init__()
        self.p, self.f = p, f
        self.validated = False

    def __call__(self, *args, **kwargs) -> Trace:
        p_out: Trace = self.p(*args, **kwargs)  # type: ignore
        f_out: Trace = auxiliary(self.f)(_output(p_out), *args, **kwargs)  # type: ignore
        p_trace, f_trace = p_out, f_out

        # NOTE: we need to walk across all nodes to see if substitution was used
        under_substitution = any(
            map(is_substituted, [*p_trace.nodes.values(), *f_trace.nodes.values()])
        )

        if not self.validated:
            assert no_samples_overlap(p_out, f_out), (
                f"{type(self)}: addresses must not overlap.\nGot:\n"
                + f"    target trace addresses: {_addrs(p_out)}\n"
                + f"    kernel trace addresses: {_addrs(f_out)}\n"
            )
            assert all(map(not_observed, f_trace.nodes.values()))
            if not under_substitution:
                assert _logweight(f_out) == 0.0
                assert all(map(not_substituted, f_trace.nodes.values()))
            self.validated = True

        log_u2 = stacked_log_prob(f_trace, site_filter=node_filter(is_sample_type))

        trace = concat_traces(p_out, f_out, site_filter=node_filter(is_sample_type))
        set_input(trace, args=args, kwargs=kwargs)
        set_param(trace, _RETURN, "return", value=_output(f_out))
        trace.nodes[_RETURN]["infer"] = {
            "m_return_node": p_out.nodes[_RETURN]
        }  # see get_marginals
        set_param(trace, _LOGWEIGHT, "return", value=_logweight(p_out) + log_u2)
        return trace


class compose(proposals):
    def __init__(self, q2: Proposals, q1: Proposals):
        super().__init__()
        self.q1, self.q2 = q1, q2
        self.validated = False

    def __call__(self, *args, **kwargs) -> Trace:
        q1_out = self.q1(*args, **kwargs)  # type: ignore
        q2_out = self.q2(q1_out.nodes[_RETURN]["value"], *args, **kwargs)  # type: ignore

        if not self.validated:
            assert no_samples_overlap(q2_out, q1_out), (
                f"{type(self)}: addresses must not overlap.\nGot:\n"
                + f"    program trace addresses: {_addrs(q1_out)}\n"
                + f"    kernel trace addresses : {_addrs(q2_out)}\n"
            )
            self.validated = True

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

        set_param(
            trace,
            _LOSS,
            "return",
            value=q1_out.nodes.get(_LOSS, {"value": 0.0})["value"],
        )

        return trace


def default_loss_fn(
    _1: Trace, _2: Trace, _3: Tensor, _4: Tensor
) -> Union[Tensor, float]:
    return 0.0


class propose(proposals):
    def __init__(
        self,
        p: Targets,
        q: Proposals,
        loss_fn: Callable[
            [Trace, Trace, Tensor, Tensor], Union[Tensor, float]
        ] = default_loss_fn,
    ):
        super().__init__()
        self.p, self.q = p, q
        self.loss_fn = loss_fn
        self.validated = False

    def _compute_logweight(
        self, p_trace: Trace, q_trace: Trace
    ) -> Tuple[Tensor, Tensor]:
        """
        compute the (decomposed) log weight. We want this to be decomposed to
        provide more information to the loss function.

        NOTE: the log weight is detached, but the incremental weight IS NOT!
        """

        lu = stacked_log_prob(
            q_trace, site_filter=sample_filter(_or(not_substituted, is_observed))
        )
        log_incremental = _logweight(p_trace) - lu

        return _logweight(q_trace).detach(), log_incremental

    def __call__(self, *args, **kwargs) -> Trace:
        q_out = self.q(*args, **kwargs)  # type: ignore
        p_out = with_substitution(self.p, trace=q_out)(*args, **kwargs)  # type: ignore

        if not self.validated:
            if no_samples_overlap(q_out, p_out) and self.loss_fn is not default_loss_fn:
                logger.warning(
                    "no overlap found between proposal and trace: no gradients will be produced"
                )
            self.validated = True

        m_trace, m_output = get_marginal(p_out)

        # FIXME local gradient computations (show how to do this in NVI example).
        # something like:
        #
        #     def rerun_as_detached_values():
        #         def rerun(loss_fn):
        #             def call(_p_trace, _q_trace, lw, lv):
        #                 p_trace = rerun_with_detached_values(_p_trace)
        #                 m_trace = rerun_with_detached_values(_m_trace)
        #                 return loss_fn(p_trace, m_trace, lw, lv)
        #             return call
        #         return rerun
        trace = (
            rerun_with_detached_values(m_trace)
            if is_nested_objective(self.loss_fn)
            else m_trace
        )

        set_input(trace, args=args, kwargs=kwargs)

        set_param(trace, _RETURN, "return", value=m_output["value"])

        lw, lv = self._compute_logweight(p_out, q_out)
        set_param(trace, _LOGWEIGHT, "return", value=lw + lv.detach())

        prev_loss = q_out.nodes.get(_LOSS, {"value": 0.0})["value"]

        accum_loss = self.loss_fn(p_out, q_out, lw, lv)  # m_trace, last time

        set_param(trace, _LOSS, "return", value=prev_loss + accum_loss)

        return trace


def is_nested_objective(fn):
    """test to see if the function was wrapped in `nested_objective`"""
    import inspect

    return inspect.getsource(fn) == inspect.getsource(nested_objective(None))


def nested_objective(loss_fn):
    """an annotation for objectives which are nested"""

    def call(_p_trace, _q_trace, lw, lv):
        p_trace = rerun_with_detached_values(_p_trace)
        q_trace = rerun_with_detached_values(_q_trace)
        loss = loss_fn(p_trace, q_trace, lw, lv)
        return loss

    return call


def rerun_with_detached_values(trace: Trace):
    newtrace = Trace()

    for name, node in trace.nodes.items():
        value = node.get("value", None)
        value = (
            value.detach()
            if isinstance(value, torch.Tensor) and value.requires_grad
            else value
        )
        newtrace.add_node(
            name, value=value, **{k: v for k, v in node.items() if k != "value"}
        )

    for p, s in zip(trace._pred, trace._succ):
        newtrace.add_edge(p, s)

    return newtrace


class augment_logweight(object):
    """
    NOTE: technically this could be a trivial function wrapper like:

        def augment(p:propose, pre:Callable):
            def doit(self, p_trace, q_trace):
                return initial_computation(self, *pre(p_trace, q_trace))
            p._compute_logweight = doit
            return p

    This class-based method is chosen to keep in line with the rest of the
    module's style (technically all classes here are just partially evaluated
    functions).

    The semantics ask for an existential, here we just "Leave No Trace" and
    return the propose instance back to its original state.
    """

    def __init__(
        self, instance: propose, pre: Callable[[Trace, Trace], Tuple[Trace, Trace]]
    ):
        if not isinstance(instance, propose):
            raise TypeError("expecting a propose instance")
        self.propose = instance

    def __call__(self, *args, **kwargs) -> Trace:
        initial_computation = self.propose._compute_logweight

        def augmented_compute_logweight(self, p_trace, q_trace):
            return initial_computation(*self.pre(p_trace, q_trace))

        self.propose._compute_logweight = augmented_compute_logweight  # type: ignore
        out = self.propose(*args, **kwargs)

        self.propose._compute_logweight = initial_computation
        return out
