# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import logging
import torch
import torch.nn.functional as F
import math
from torch import Tensor, tensor
from typing import Any, Callable, Tuple, Set, TypeVar, Union, Optional, Tuple, Sequence

from pyro.poutine import Trace
from pyro.distributions.distribution import Distribution
from pyro.poutine.handlers import _make_handler
from pyro.poutine.messenger import Messenger
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.trace_messenger import TraceMessenger

logger = logging.getLogger(__name__)

# type aliases

Node = dict
T = TypeVar("T")
Predicate = Callable[[Any], bool]
SiteFilter = Callable[[str, Node], bool]


def flatmap_trace(
    *traces: Trace,
    site_map: Callable[[str, dict], dict] = lambda name, node: node,
    site_filter: Callable[[str, dict], bool] = lambda name, node: True,
) -> Trace:
    newtrace = Trace()
    for tr in traces:
        drop_edges = []
        for name, node in tr.nodes.items():
            if site_filter(name, node):
                newtrace.add_node(name, **site_map(name, node))
            else:
                drop_edges.append(name)
        for l, r in tr.edges:
            if not (l in drop_edges and r in drop_edges):
                newtrace.add_edge(l, r)

    return newtrace


def concat_traces(
    *traces: Trace,
    site_filter: Callable[[str, Any], bool] = lambda name, node: True,
) -> Trace:
    return flatmap_trace(*traces, site_filter=site_filter)


def no_samples_overlap(t0: Trace, t1: Trace):
    return len(_addrs(t0).intersection(_addrs(t1))) == 0


def is_observed(node: Node) -> bool:
    return node.get("is_observed", False)


def is_sample_type(node: Node) -> bool:
    return node["type"] == "sample"


def is_substituted(node: Node):
    return node.get("infer", dict()).get("substituted", False)


def is_auxiliary(node: Node):
    return node.get("infer", dict()).get("is_auxiliary", False)


_RETURN, _LOGWEIGHT, _LOSS = "_RETURN", "_LOGWEIGHT", "_LOSS"


def set_input(trace, args, kwargs):
    trace.add_node("_INPUT", name="_INPUT", type="input", args=args, kwargs=kwargs)


def set_param(trace, name, type, value=None):
    trace.add_node(name, name=name, type=type, value=value)


def valueat(tr, a):
    return tr.nodes[a]["value"]


def _addrs(tr):
    return {k for k, n in tr.nodes.items() if is_sample_type(n)}


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
    m_trace = concat_traces(
        trace, site_filter=lambda _, n: is_sample_type(n) and not is_auxiliary(n)
    )
    return m_trace, m_output


class inference(object):
    pass


Inference = Union[inference, Callable[..., Trace]]


class targets(inference):
    pass


Targets = Union[targets, Callable[..., Trace]]


class proposals(inference):
    # FIXME: debug and q_prev should be moved to a trace aggregation effect
    debug = None
    q_prev = None
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
            lp = stacked_log_prob(
                tr,
                lambda _, n: is_sample_type(n)
                and (is_substituted(n) or is_observed(n)),
            )

            trace = tr
            set_input(trace, args=args, kwargs=kwargs)
            set_param(trace, _RETURN, "return", value=out)
            set_param(trace, _LOGWEIGHT, "return", value=lp)
            return trace

    def __repr__(self) -> str:
        return f"<{type(self).__name__} wrapper of {repr(self.program)} at {hex(id(self))}>"


Primitive = Union[Callable[..., Trace], primitive]


class extend(targets):
    def __init__(self, p: Targets, f: Primitive):
        super().__init__()
        self.p, self.f = p, f
        self.validated = False

    def __call__(self, *args, **kwargs) -> Trace:
        p_out: Trace = self.p(*args, **kwargs)  # type: ignore
        f_out: Trace = auxiliary(self.f)(valueat(p_out, _RETURN), *args, **kwargs)  # type: ignore
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
            assert all(map(lambda n: not is_observed(n), f_trace.nodes.values()))
            if not under_substitution:
                assert valueat(f_out, _LOGWEIGHT) == 0.0
                assert all(map(lambda n: not is_substituted(n), f_trace.nodes.values()))
            self.validated = True

        log_u2 = stacked_log_prob(f_trace, site_filter=lambda _, n: is_sample_type(n))

        trace = concat_traces(p_out, f_out, site_filter=lambda _, n: is_sample_type(n))
        set_input(trace, args=args, kwargs=kwargs)
        set_param(trace, _RETURN, "return", value=valueat(f_out, _RETURN))
        trace.nodes[_RETURN]["infer"] = {
            "m_return_node": p_out.nodes[_RETURN]
        }  # see get_marginals
        set_param(
            trace, _LOGWEIGHT, "return", value=valueat(p_out, _LOGWEIGHT) + log_u2
        )
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

        trace = concat_traces(
            q2_out, q1_out, site_filter=lambda _, n: is_sample_type(n)
        )
        set_input(
            trace,
            args=args,
            kwargs=kwargs,
        )
        set_param(trace, _RETURN, "return", value=valueat(q2_out, _RETURN))

        set_param(
            trace,
            _LOGWEIGHT,
            "return",
            value=valueat(q1_out, _LOGWEIGHT) + valueat(q2_out, _LOGWEIGHT),
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
        q_prev=None,
    ):
        super().__init__()
        self.p, self.q = p, q
        self.loss_fn = loss_fn
        self.validated = False
        self.validated_lw = False
        self.q_prev = q_prev
        self.debug = None

    def _compute_logweight(
        self, p_trace: Trace, q_trace: Trace
    ) -> Tuple[Tensor, Tensor]:
        """
        compute the (decomposed) log weight. We want this to be decomposed to
        provide more information to the loss function.

        NOTE: the log weight is detached, but the incremental weight /is not/!
        """
        in_rho = lambda n: is_sample_type(n)
        in_tau = lambda n: is_sample_type(n) and not is_observed(n)
        rho_1 = {a for a, n in q_trace.nodes.items() if in_rho(n)}
        tau_1 = {a for a, n in q_trace.nodes.items() if in_tau(n)}
        tau_2 = {a for a, n in p_trace.nodes.items() if in_tau(n)}
        nodes = rho_1 - (tau_1 - tau_2)

        lu = stacked_log_prob(q_trace, site_filter=lambda a, _: a in nodes)
        lv = valueat(p_trace, _LOGWEIGHT) - lu
        lw = valueat(q_trace, _LOGWEIGHT).detach()

        if not self.validated_lw:
            # FIXME: put this in a test
            q_lps = stacked_log_prob(
                q_trace, site_filter=lambda a, n: is_sample_type(n)
            )
            p_lps = stacked_log_prob(
                p_trace, site_filter=lambda a, n: is_sample_type(n)
            )
            lv_check = p_lps - q_lps
            isclose = torch.isclose(lv, lv_check)
            assert isclose.all().item(), "incremental weight is constructed correctly"
            self.validated_lw = True

        log_weight = lw
        log_incremental = lv
        return log_weight, log_incremental

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

        # NOTE: here we sever local gradient computations for nested objectives.
        # NOTE: alternative is to add information into static proposal, gradients must be severed in the output trace
        trace = detach_values(m_trace) if is_nested_objective(self.loss_fn) else m_trace

        set_input(trace, args=args, kwargs=kwargs)

        set_param(trace, _RETURN, "return", value=m_output["value"])

        lw, lv = self._compute_logweight(p_out, q_out)
        set_param(trace, _LOGWEIGHT, "return", value=lw + lv.detach())

        prev_loss = q_out.nodes.get(_LOSS, {"value": 0.0})["value"]

        # NOTE: this only required m_trace in the first codebase
        accum_loss = self.loss_fn(p_out, q_out, lw, lv)

        set_param(trace, _LOSS, "return", value=prev_loss + accum_loss)

        if self.q_prev is not None:
            self.debug = dict(
                out=trace, q_out=q_out, p_out=p_out, prev=self.q_prev.debug
            )

        return trace


def is_nested_objective(fn):
    """test to see if the function was wrapped in `nested_objective`"""
    import inspect

    return inspect.getsource(fn) == inspect.getsource(nested_objective(None))


def nested_objective(loss_fn):
    """an annotation for objectives which are nested"""

    def call(_p_trace, _q_trace, lw, lv):
        out_names = {
            k
            for k, v in _p_trace.nodes.items()
            if is_sample_type(v) and not is_auxiliary(v)
        }
        # p_trace, q_trace = detach_values(_p_trace, site_filter=lambda k, v: k in out_names), \
        #    detach_values(_q_trace, site_filter=lambda k, v: k in out_names)
        # p_trace, q_trace = detach_values(_p_trace, site_filter=lambda k, v: k in out_names), _q_trace
        p_trace, q_trace = _p_trace, detach_values(
            _q_trace, site_filter=lambda k, v: k in out_names
        )
        # p_trace, q_trace = _p_trace, _q_trace
        loss = loss_fn(p_trace, q_trace, lw, lv)
        return loss

    return call


def detach_values(trace: Trace, site_filter=lambda a, b: True):
    newtrace = Trace()

    detachit = (
        lambda v: v.detach() if isinstance(v, torch.Tensor) and v.requires_grad else v
    )
    keys = ["value"]

    for name, node in trace.nodes.items():
        value = node.get("value", None)
        detached = dict(value=value)
        if site_filter(name, node):
            detached = dict(value=detachit(value))
            if "log_prob" in node:
                del node["log_prob"]
            if "unscaled_log_prob" in node:
                del node["unscaled_log_prob"]

        newtrace.add_node(
            name, **detached, **{k: v for k, v in node.items() if k not in keys}
        )

    for p, s in zip(trace._pred, trace._succ):
        newtrace.add_edge(p, s)

    return newtrace


class augment_logweight(object):
    """
    NOTE: technically this could be a trivial function wrapper like:

        def augment(p:propose, pre:Callable):
            initial_computation = p._compute_logweight
            def doit(self, p_trace, q_trace):
                return initial_computation(self, *pre(p_trace, q_trace))
            p._compute_logweight = doit
            return p

    This class-based method is chosen to keep in line with the rest of the
    module's style (technically all classes here are just partially evaluated
    functions).

    The semantics ask for an existential type, instead we "Leave No Trace" and
    return the propose instance back to its original state.
    """

    def __init__(
        self, instance: propose, pre: Callable[[Trace, Trace], Tuple[Trace, Trace]]
    ):
        if not isinstance(instance, propose):
            raise TypeError("expecting a propose instance")
        self.propose = instance
        self.pre = pre

    def __call__(self, *args, **kwargs) -> Trace:
        initial_computation = self.propose._compute_logweight

        def augmented_compute_logweight(p_trace, q_trace):
            return initial_computation(*self.pre(p_trace, q_trace))

        self.propose._compute_logweight = augmented_compute_logweight  # type: ignore
        out = self.propose(*args, **kwargs)

        self.propose._compute_logweight = initial_computation
        return out


def _is_dist_node(n: dict):
    return "fn" in n.keys() and isinstance(n["fn"], Distribution)


def _dist_with_detached_args(node):
    if not _is_dist_node(node):
        raise ValueError("Trace node type is not supported")

    dist = node["fn"]

    node_detached = node.copy()
    import inspect

    sig = inspect.signature(dist.__class__.__init__)
    # FIXME: pretty sure we need to recurse into each density for Tempered
    node_detached["fn"] = dist.__class__(
        **{
            p.name: dist.__dict__[p.name].detach()
            if isinstance(dist.__dict__[p.name], Tensor)
            else dist.__dict__[p.name]
            for p in list(sig.parameters.values())[1:]
            if p.name in dist.__dict__
        }
    )
    if getattr(node_detached["fn"], "has_rsample", False):
        node_detached["fn"].has_rsample = False
        assert not node_detached[
            "fn"
        ].has_rsample, f"setting has_rsample does not stick on {dist.__class__}"

    node_detached["value"] = node["value"]
    return node_detached


def stl_trick(p_trace, q_trace):
    """
    adjust trace to compute a "sticking the landing" (stl) gradient.
    """
    q_stl_trace = concat_traces(
        q_trace, site_filter=lambda _, n: not _is_dist_node(n)
    )  # generate a new trace
    dist_nodes = {k for k, n in q_trace.nodes.items() if _is_dist_node(n)}

    for k in dist_nodes:
        dereparameterized = _dist_with_detached_args(q_trace.nodes[k])
        q_stl_trace.add_node(k, **dereparameterized)

    for l, r in q_trace.edges:
        if l in dist_nodes or r in dist_nodes:
            q_stl_trace.add_edge(l, r)

    return (p_trace, q_stl_trace)


# ============================================
# resampling
# ============================================


def ancestor_indices_systematic(lw, sample_dims, batch_dim):
    assert batch_dim is not None and sample_dims is not None
    _sample_dims = -1
    n, b = lw.shape[sample_dims], lw.shape[batch_dim]

    u = torch.rand(b, device=lw.device)
    usteps = torch.stack([(k + u) for k in range(n)], dim=_sample_dims) / n
    nws = F.softmax(lw.detach(), dim=sample_dims)

    csum = nws.transpose(sample_dims, _sample_dims).cumsum(dim=_sample_dims)
    cmax, _ = torch.max(csum, dim=_sample_dims, keepdim=True)
    ncsum = csum / cmax

    aidx = torch.searchsorted(ncsum, usteps, right=False)
    aidx = aidx.transpose(_sample_dims, sample_dims)
    return aidx


def _pick(z, aidx, sample_dims):
    ddim = z.dim() - aidx.dim()

    assert (
        z.shape[: z.dim() - ddim] == aidx.shape
    ), "data dims must be at the end of arg:z"

    mask = aidx
    for _ in range(ddim):
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(z)

    return z.gather(sample_dims, mask)


def _pick_node(node: dict, ancestor_indicies: Tensor, dim=0):
    """
    Draws ``num_samples`` samples from ``input`` at dimension ``dim``.

    :param torch.Tensor input: the input tensor.
    :param int num_samples: the number of samples to draw from ``input``.
    :param int dim: dimension to draw from ``input``.
    :returns torch.Tensor: resampled ``input`` using systematic resampling.
    """
    assert "value" in node

    # FIXME: Do not detach all
    value = _pick(node["value"], ancestor_indicies, sample_dims=dim)
    log_prob = (
        _pick(node["_computed_log_weights"], ancestor_indicies, sample_dims=dim)
        if "_computed_log_weights" in node
        else None
    )

    ret = node.copy()
    ret["value"] = value
    if log_prob is not None:
        ret["_computed_log_weights"] = log_prob

    return ret


def resample_trace_systematic(
    trace: Trace,
    sample_dims: int,  # FIXME: review that this is, indeed an int. if this is a tuple, need more verification
    batch_dim: Optional[int],
    normalize_weights: bool = False,
) -> Trace:
    assert _LOGWEIGHT in trace.nodes
    log_weight = trace.nodes[_LOGWEIGHT]["value"]

    _batch_dim = None
    if batch_dim is None:
        lwsize = len(log_weight.shape)
        assert lwsize != 2, "Found a undeclared batch dim on log_weight."
        assert len(log_weight.shape) == 1, "batch_dim None requires 1d log_weight"
        _batch_dim = (sample_dims + 1) % 2  # expand the other dimension
        log_weight = log_weight.unsqueeze(_batch_dim)

    aidx = ancestor_indices_systematic(
        log_weight,
        sample_dims=sample_dims,
        batch_dim=_batch_dim if batch_dim is None else batch_dim,
    )

    if batch_dim is None:
        aidx = aidx.squeeze(_batch_dim)

    # If we are resampling the output of a kernel function, the return may be a
    # resampled output. We keep track of this here:
    ret = trace.nodes.get(_RETURN, {"value": None})

    def fixme_equality(o, n):
        if isinstance(o, Tensor) and isinstance(n, Tensor):
            return o is n or (o.shape == n.shape and torch.equal(o, n))
        else:
            return o is n

    def _resample(_, site):
        if (  # FIXME: triple check that these semantics line up with the paper.
            # especially compatibility with pyro-based semantics
            # do not resample unsamplable sites
            site["type"] != "sample"
            # do not resample observed distributions
            or site["is_observed"]
            # do not resample if it is not a pyro distribution
            or not isinstance(site["fn"], Distribution)
            # do not resample if it is a result of a plate effect
            or isinstance(site["fn"], _Subsample)
        ):
            return site
        else:
            newsite = _pick_node(site, aidx, dim=sample_dims)

            # FIXME: flatmap_trace might be doing the "right thing" and this can be removed. Test this.

            # ensure that the resampled site will change a model's output:
            if fixme_equality(site["value"], ret["value"]):
                ret["value"] = newsite["value"]
            elif isinstance(ret["value"], dict):
                for k, v in ret["value"].items():  # type: ignore
                    if fixme_equality(site["value"], v):
                        ret["value"][k] = newsite["value"]  # type: ignore
            elif isinstance(ret["value"], Sequence):
                for i, v in enumerate(ret["value"]):  # type: ignore
                    if fixme_equality(site["value"], v):
                        ret["value"][i] = newsite["value"]  # type: ignore

            return newsite

    new_trace = flatmap_trace(
        trace, site_map=_resample, site_filter=lambda k, _: k != _LOGWEIGHT
    )

    log_weight_node = trace.nodes[
        _LOGWEIGHT
    ].copy()  # NOTE: expensive, can we remove this?
    log_weight = log_weight_node["value"]
    log_weight = torch.logsumexp(
        log_weight - math.log(log_weight.shape[sample_dims]),
        dim=sample_dims,
        keepdim=True,
    ).expand_as(log_weight)

    if normalize_weights:
        log_weight = F.softmax(log_weight, dim=sample_dims).log()

    log_weight_node["value"] = log_weight
    new_trace.add_node(_LOGWEIGHT, **log_weight_node)

    return new_trace


class resample(proposals):
    """
    the resample combinator is a proposal which applies systematic resampling on
    every site in the trace.
    """

    def __init__(
        self,
        q: Proposals,
        sample_dim: int,
        batch_dim: Optional[int] = None,
        normalize_weights: bool = False,
    ):
        super().__init__()
        self.q = q
        self.sample_dim, self.batch_dim = sample_dim, batch_dim
        self.normalize_weights = normalize_weights

    def __call__(self, *args, **kwargs) -> Trace:
        trace = self.q(*args, **kwargs)  # type: ignore

        return resample_trace_systematic(
            trace, self.sample_dim, self.batch_dim, self.normalize_weights
        )
