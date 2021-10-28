import torch
from torch import Tensor, tensor
from typing import NamedTuple, Any, Callable, Union, TypeVar, Set

from pyro.poutine import Trace
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.trace_messenger import TraceMessenger
from pyro.poutine.messenger import Messenger
from pyro.poutine.handlers import _make_handler


# type aliases

Node = dict
T = TypeVar("T")
Predicate = Callable[[Any], bool]
SiteFilter = Callable[[str, Node], bool]


def concat_traces(
    *traces: Trace, site_filter: Callable[[str, Any], bool] = lambda name, node: True
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


def assert_no_overlap(t0: Trace, t1: Trace, location=""):
    assert (
        len(set(t0.nodes.keys()).intersection(set(t1.nodes.keys()))) == 0
    ), f"{location}: addresses must not overlap"


def is_observed(node: Node) -> bool:
    return node["is_observed"]


def not_observed(node: Node) -> bool:
    return not is_observed(node)


def _check_infer_map(k: str) -> Callable[[Node], bool]:
    return lambda node: k in node["infer"] and node["infer"][k]


is_substituted = _check_infer_map("substituted")
is_auxiliary = _check_infer_map("is_auxiliary")


def not_substituted(node: Node) -> bool:
    return not is_substituted(node)


def _or(p0: Predicate, p1: Predicate) -> Predicate:
    return lambda x: p0(x) or p1(x)


def node_filter(p: Callable[[Node], bool]) -> SiteFilter:
    return lambda _, node: p(node)


def addr_filter(p: Callable[[str], bool]) -> SiteFilter:
    return lambda name, _: p(name)


def membership_filter(members: Set[str]) -> SiteFilter:
    return lambda name, _: name in members


class Out(NamedTuple):
    output: Any
    log_weight: Tensor
    trace: Trace


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
    return concat_traces(trace, site_filter=lambda _, n: not is_auxiliary(n))


class inference(object):
    def __init__(self):
        self.loss = 0.0


Inference = Union[inference, Callable[..., Out]]


class targets(inference):
    def __init__(self):
        super().__init__()


Targets = Union[targets, Callable[..., Out]]


class proposals(inference):
    def __init__(self):
        super().__init__()


Proposals = Union[proposals, Callable[..., Out]]

def stacked_log_prob(trace, site_filter: Callable[[str, Any], bool] = lambda name, node: True):
    trace.compute_log_prob(site_filter=site_filter)
    log_probs = [n['log_prob'] for k, n in trace.nodes.items() if 'log_prob' in n and site_filter(k, n)]
    log_prob  = torch.stack(log_probs).sum(dim=0) if len(log_probs) > 0 else tensor([0.])
    if len(log_prob.shape) == 0:
        log_prob.unsqueeze_(0)
    return log_prob

class primitive(targets, proposals):
    def __init__(self, program: Callable[..., Any]):
        super().__init__()
        self.program = program

    def __call__(self, *args, **kwargs) -> Out:
        with TraceMessenger() as tracer:
            out = self.program(*args, **kwargs)
            tr: Trace = tracer.trace
            lp = stacked_log_prob(tr, node_filter(_or(is_substituted, is_observed)))
            # FIXME: put out on the trace as well so that these are correctly plated
            return Out(output=out, log_weight=lp, trace=tr)


Primitive = Union[Callable[..., Out], primitive]


class extend(targets):
    def __init__(self, p: Targets, f: Primitive):
        super().__init__()
        self.p, self.f = p, f

    def __call__(self, *args, **kwargs) -> Out:
        p_out: Out = self.p(*args, **kwargs)
        f_out: Out = auxiliary(self.f)(p_out.output, *args, **kwargs)
        p_trace, f_trace = p_out.trace, f_out.trace

        # NOTE: we need to walk across all nodes to see if substitution was used
        under_substitution = any(
            map(is_substituted, [*p_trace.nodes.values(), *f_trace.nodes.values()])
        )

        assert_no_overlap(p_trace, f_trace, location=type(self))
        assert all(map(not_observed, f_trace.nodes.values()))
        if not under_substitution:
            assert f_out.log_weight == 0.0
            assert all(map(not_substituted, f_trace.nodes.values()))

        log_u2 = stacked_log_prob(f_trace)
        return Out(
            trace=concat_traces(p_out.trace, f_out.trace),
            log_weight=p_out.log_weight + log_u2,
            output=f_out.output,
        )


class compose(proposals):
    def __init__(self, q2: Proposals, q1: Proposals):
        super().__init__()
        self.q1, self.q2 = q1, q2

    def __call__(self, *args, **kwargs) -> Out:
        q1_out = self.q1(*args, **kwargs)
        q2_out = self.q2(*args, **kwargs)
        assert_no_overlap(q2_out.trace, q1_out.trace, location=type(self))

        return Out(
            trace=concat_traces(q2_out.trace, q1_out.trace),
            log_weight=q1_out.log_weight + q2_out.log_weight,
            output=q2_out.output,
        )


class propose(proposals):
    def __init__(
        self,
        p: Targets,
        q: Proposals,
        loss_fn: Callable[[Out], Union[Tensor, float]] = (lambda out: 0.0),
    ):
        super().__init__()
        self.p, self.q = p, q
        self.loss_fn = loss_fn

    def __call__(self, *args, **kwargs) -> Out:
        q_out = self.q(*args, **kwargs)
        p_out = with_substitution(self.p, trace=q_out.trace)(*args, **kwargs)

        # all_nodes = set(q_out.trace.nodes.keys())
        # sampled_nodes = set({k for k, v in q_out.trace.nodes.items() if not_observed(v)})
        # substituted_nodes = set({k for k, v in p_out.trace.nodes.items() if not_observed(v)})
        # lu_1 = q_out.trace.log_prob_sum(membership_filter(all_nodes - (sampled_nodes - substituted_nodes)))

        # FIXME this hook exists to alter an NVI trace for stl
        # q_trace = dispatch(self.transf_q_trace, q_out.trace, **inf_kwargs)

        lu_1 = stacked_log_prob(q_out.trace, node_filter(_or(not_substituted, is_observed)))
        lw_1 = q_out.log_weight

        # We call that lv because its the incremental weight in the IS sense
        lv = p_out.log_weight - lu_1
        lw_out = lw_1 + lv

        m_trace = get_marginal(p_out.trace)
        # FIXME: this is not accounted for -- will return the final kernel output, not the initial output
        # should be something like: m_output = m_trace["_RETURN"]
        m_output = p_out.output
        self.loss = self.loss + self.loss_fn(m_output)

        return Out(
            # FIXME local gradient computations
            # trace=m_trace if self._no_reruns else rerun_with_detached_values(m_trace),
            trace=m_trace,
            log_weight=lw_out.detach(),
            output=m_output,
        )
