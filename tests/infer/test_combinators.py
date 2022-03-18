# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from operator import itemgetter
from typing import Callable, Union

import pytest
import torch
from pytest import mark
from torch import Tensor, tensor

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.combinators import (
    Node,
    Trace,
    compose,
    extend,
    get_marginal,
    is_auxiliary,
    primitive,
    propose,
    with_substitution,
    Set,
    SiteFilter,
    _RETURN,
)
from pyro.poutine import Trace, replay


def node_filter(p: Callable[[Node], bool]) -> SiteFilter:
    return lambda _, node: p(node)


def addr_filter(p: Callable[[str], bool]) -> SiteFilter:
    return lambda name, _: p(name)


def membership_filter(members: Set[str]) -> SiteFilter:
    return lambda name, _: name in members


def seed(s=42) -> None:
    import random

    import numpy as np

    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    # just incase something goes wrong with set_deterministic
    torch.backends.cudnn.benchmark = True
    if torch.__version__[:3] == "1.8":
        pass
        # torch.use_deterministic_algorithms(True)


def tensor_of(n: Union[float, Tensor]) -> Tensor:
    return torch.ones(1) * n


@pytest.fixture(scope="session", autouse=True)
def simple1():
    def model():
        z_1 = pyro.sample("z_1", dist.Normal(tensor_of(1), tensor_of(1)))
        z_2 = pyro.sample("z_2", dist.Normal(tensor_of(2), tensor_of(2)))

        pyro.sample("x_1", dist.Normal(z_1, tensor_of(1)), obs=z_1)
        pyro.sample("x_2", dist.Normal(z_2, tensor_of(2)), obs=z_2)

        return tensor_of(3)

    yield model


@pytest.fixture(scope="session", autouse=True)
def simple2():
    def model():
        z_2 = pyro.sample("z_2", dist.Normal(tensor_of(2), tensor_of(2)))
        z_3 = pyro.sample("z_3", dist.Normal(tensor_of(3), tensor_of(3)))

        pyro.sample("x_2", dist.Normal(tensor_of(2), tensor_of(2)), obs=z_2)
        pyro.sample("x_3", dist.Normal(tensor_of(3), tensor_of(3)), obs=z_3)

        return tensor_of(1)

    yield model


@pytest.fixture(scope="session", autouse=True)
def simple3():
    def model(c):
        z_3 = pyro.sample("z_3", dist.Normal(tensor_of(3), tensor_of(3)))

        pyro.sample("x_3", dist.Normal(tensor_of(3), tensor_of(3)), obs=z_3)

        return None

    yield model


@pytest.fixture(scope="session", autouse=True)
def simple4():
    def model(c):
        pyro.sample("z_1", dist.Normal(tensor_of(c), tensor_of(c)))
        return c + 1

    yield model


@pytest.fixture(scope="session", autouse=True)
def simple5():
    def model(c):
        pyro.sample("z_5", dist.Normal(tensor_of(c), tensor_of(c)))
        return c + 1

    yield model


# ===========================================================================
# Models
@pytest.fixture
def model0():
    def model():
        cloudy = pyro.sample("cloudy", dist.Bernoulli(0.3))
        cloudy = "cloudy" if cloudy.item() == 1.0 else "sunny"
        mean_temp = {"cloudy": 55.0, "sunny": 75.0}[cloudy]
        scale_temp = {"cloudy": 10.0, "sunny": 15.0}[cloudy]
        temp = pyro.sample("temp", dist.Normal(mean_temp, scale_temp))
        return cloudy, temp.item()

    yield model


@pytest.fixture
def model1():
    def model():
        flip = pyro.sample("flip", dist.Bernoulli(0.3))
        return "heads" if flip.item() == 1.0 else "tails"

    yield model


# ===========================================================================
# Asserts
def assert_no_observe(model):
    assert isinstance(model, primitive)

    trace = poutine.trace(model).get_trace()
    for name, node in trace.nodes.items():
        if node["type"] == "sample":
            assert not node["is_observed"], f"node {name} is observed!"


def assert_no_overlap(primitive_model, non_overlapping_primitive):
    tr0, tr1 = primitive_model(), non_overlapping_primitive()
    tr0_names = {name for name, site in tr0.nodes.items() if site["type"] == "sample"}
    for name, _ in tr1.nodes.items():
        assert name not in tr0_names, f"{name} is in both traces!"


def assert_log_weight_zero(primitive_output):
    lw = primitive_output.nodes["_LOGWEIGHT"]["value"]
    assert isinstance(lw, Tensor) and lw == 0.0, f"log weight is not zero! Got: {lw}"


# primitive expression tests
# ===========================================================================


def test_constructor(model0):
    m = primitive(model0)
    assert_no_observe(m)
    out = m()
    assert_no_observe(m)
    assert_log_weight_zero(out)

    o = out.nodes["_RETURN"]["value"][0]
    assert isinstance(o, str) and o in {"cloudy", "sunny"}
    assert isinstance(out, Trace)


def test_no_overlapping_variables(model0, model1):
    m0 = primitive(model0)
    m1 = primitive(model1)
    assert_no_observe(m0)
    assert_no_observe(m1)
    assert_no_overlap(m0, m1)


def valueat(o, a):
    return o.nodes[a]["value"]


def _log_weight(tr):
    return valueat(tr, "_LOGWEIGHT")


def _addrs(tr):
    return {k for k, n in tr.nodes.items() if n["type"] == "sample"}


def test_with_substitution_simple(model0):
    p = primitive(model0)
    q = primitive(model0)
    p_out = p()
    q_out = with_substitution(q, trace=p_out)()

    p_addrs = _addrs(p_out)
    q_addrs = _addrs(q_out)
    assert p_addrs.intersection(q_addrs) == p_addrs.union(q_addrs)

    for a in p_addrs:
        if q_out.nodes[a]["type"] == "sample":
            assert q_out.nodes[a]["value"] == p_out.nodes[a]["value"]
            assert valueat(q_out, a) == valueat(p_out, a)
            assert q_out.nodes[a]["infer"]["substituted"] is True

    assert valueat(p_out, "_RETURN") == valueat(
        q_out, "_RETURN"
    ), f"return values should be equal"
    assert (
        valueat(q_out, "_LOGWEIGHT") != 0.0
    ), "log weight is zero, was expecting non-zero value"


def test_with_substitution_and_plates():
    def model(data):
        """example from SVI pt. 1"""
        alpha0 = torch.tensor(10.0)
        beta0 = torch.tensor(10.0)
        f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
        for i in range(len(data)):
            pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

    inp = torch.tensor([i % 2 for i in range(11)]).double()

    with pyro.plate("data1", 3), pyro.plate("data2", 5), pyro.plate("data3", 7):
        p = primitive(model)
        p_out = p(inp)

    assert _log_weight(p_out).shape == torch.Size(
        [7, 5, 3]
    ), "plated calls should return weights for independent dimensions"

    with pyro.plate("data", 3):
        q = primitive(model)
        q_out = q(inp)

        p = primitive(model)
        p_out = with_substitution(p, trace=q_out)(inp)

    p_addrs, q_addrs = _addrs(p_out), _addrs(q_out)
    assert p_addrs.intersection(q_addrs) == p_addrs.union(q_addrs)

    for a in p_addrs:
        pnode = p_out.nodes[a]
        qnode = q_out.nodes[a]
        pval, qval = pnode.get("value", torch.tensor([float("NaN")])), qnode.get(
            "value", torch.tensor([float("NaN")])
        )
        assert torch.equal(
            pval, qval
        ), f"expected identical '{a}' field, got: {pval} vs. {qval}"
        assert pnode["is_observed"] ^ pnode["infer"].get("substituted", False)

    assert valueat(p_out, "_RETURN") == valueat(q_out, "_RETURN")
    assert torch.all(
        torch.ne(_log_weight(q_out), torch.zeros(_log_weight(q_out).shape))
    ).item()


def starts_with_x(n):
    return n[0] == "x"


# test simple programs
def test_simple1(simple1):
    s1_out = primitive(simple1)()
    assert _addrs(s1_out) == {"z_1", "z_2", "x_1", "x_2"}
    assert _log_weight(s1_out) == s1_out.log_prob_sum(addr_filter(starts_with_x))


def test_simple2(simple2):
    out = primitive(simple2)()
    assert _addrs(out) == {"x_2", "x_3", "z_2", "z_3"}
    assert _log_weight(out) == out.log_prob_sum(addr_filter(starts_with_x))
    assert _log_weight(out) != 0.0


def test_simple3(simple3):
    out = primitive(simple3)(None)
    assert _addrs(out) == {"z_3", "x_3"}
    assert _log_weight(out) == out.log_prob_sum(addr_filter(starts_with_x))


def test_simple4(simple4):
    out = primitive(simple4)(tensor([1]))
    assert _addrs(out) == {"z_1"}
    assert _log_weight(out) == 0.0


def test_simple_substitution(simple1, simple2):
    s1, s2 = primitive(simple1), primitive(simple2)
    s1_out = s1()
    s2_out = with_substitution(s2, trace=s1_out)()

    rho_f_addrs = {"x_2", "x_3", "z_2", "z_3"}
    tau_f_addrs = {"z_2", "z_3"}
    tau_p_addrs = {"z_1", "z_2"}
    nodes = rho_f_addrs - (tau_f_addrs - tau_p_addrs)
    lw_out = s2_out.log_prob_sum(membership_filter(nodes))

    assert (lw_out == _log_weight(s2_out)).all()
    assert len(set({"z_1", "x_1"}).intersection(_addrs(s2_out))) == 0


def test_extend_unconditioned_no_plates(simple2, simple4):
    s2, s4 = primitive(simple2), primitive(simple4)

    p_out = s2()

    replay_s2 = poutine.replay(s2, trace=p_out)
    tau_2 = {"z_2", "z_3", "x_2", "x_3"}
    assert _addrs(p_out) == tau_2
    assert _log_weight(p_out) == batched_log_prob_sum(p_out, addr_filter(starts_with_x))

    f_out = s4(valueat(p_out, _RETURN))
    f_addrs = _addrs(f_out)
    replay_s4 = poutine.replay(s4, trace=f_out)
    tau_star = {"z_1"}
    assert _addrs(f_out) == tau_star
    assert _log_weight(f_out) == 0.0

    out = extend(p=replay_s2, f=replay_s4)()
    assert _addrs(out) == {"x_2", "x_3", "z_2", "z_3", "z_1"}
    assert _log_weight(out) == _log_weight(p_out) + f_out.log_prob_sum()

    def in_keys(out):
        def go(kv):
            return kv[0] in out.nodes.keys()

        return go

    p_nodes = list(filter(in_keys(p_out), out.nodes.items()))
    assert len(p_nodes) > 0
    assert all(list(map(lambda n: not is_auxiliary(n), map(itemgetter(1), p_nodes))))

    f_nodes = list(filter(lambda kv: kv[0] in f_addrs, out.nodes.items()))
    assert len(f_nodes) > 0
    assert all(list(map(is_auxiliary, map(itemgetter(1), f_nodes))))

    # Test extend log_weight
    trace_addrs = _addrs(out)
    m_trace_addrs = set(get_marginal(out)[0].nodes.keys())
    assert m_trace_addrs == tau_2

    aux_trace_addrs = trace_addrs - m_trace_addrs
    assert aux_trace_addrs == tau_star

    lw_2 = out.log_prob_sum(membership_filter({"x_2", "x_3"})) + out.log_prob_sum(
        membership_filter({"z_1"})
    )
    assert lw_2 == _log_weight(out), "_log_weight(p_out) is not expected lw_2"


def tensor_eq(t0: Tensor, t1: Tensor) -> bool:
    if t0.shape != t1.shape:
        raise AssertionError(f"shape mismatch: {t0.shape} vs {t1.shape}")
    return bool(torch.all(torch.eq(t0, t1)).item())


def batched_log_prob_sum(
    o: Trace, site_filter: Callable[[str, Node], bool] = (lambda a, b: True)
) -> Tensor:
    o.compute_log_prob(site_filter=site_filter)
    lp = torch.stack(
        [
            n["log_prob"]
            for k, n in o.nodes.items()
            if n["type"] == "sample" and site_filter(k, n)
        ]
    ).sum(dim=0)
    return lp if len(lp.shape) > 0 else lp.unsqueeze(0)


def test_extend_unconditioned_with_plates(simple2, simple4):
    s2, s4 = primitive(simple2), primitive(simple4)

    with pyro.plate("batch", 3), pyro.plate("samples", 5):
        p_out = s2()
        replay_s2 = poutine.replay(s2, trace=p_out)
        f_out = s4(valueat(p_out, _RETURN))
        replay_s4 = poutine.replay(s4, trace=f_out)

    tau_2 = {"z_2", "z_3", "x_2", "x_3"}
    assert _addrs(p_out) == tau_2
    assert tensor_eq(
        _log_weight(p_out), batched_log_prob_sum(p_out, lambda k, _: k[0] == "x")
    )
    tau_star = {"z_1"}
    assert _addrs(f_out) == tau_star
    assert (
        _log_weight(f_out).shape == torch.Size([1]) and _log_weight(f_out) == 0.0
    ), "kernel's log weight is scalar of zero"

    with pyro.plate("batch", 3), pyro.plate("samples", 5):
        out = extend(p=replay_s2, f=replay_s4)()

    assert _addrs(out) == {"x_2", "x_3", "z_2", "z_3", "z_1"}

    assert tensor_eq(_log_weight(out), _log_weight(p_out) + batched_log_prob_sum(f_out))

    p_nodes = list(filter(lambda kv: kv[0] in p_out.nodes.keys(), out.nodes.items()))
    assert len(p_nodes) > 0
    assert all(list(map(lambda kv: not is_auxiliary(kv[1]), p_nodes)))

    f_addrs = _addrs(f_out)
    f_nodes = list(filter(lambda kv: kv[0] in f_addrs, out.nodes.items()))
    assert len(f_nodes) > 0
    assert all(list(map(lambda kv: is_auxiliary(kv[1]), f_nodes)))

    # Test extend log_weight
    trace_addrs = _addrs(out)
    m_trace_addrs = set(get_marginal(out)[0].nodes.keys())
    assert m_trace_addrs == tau_2

    aux_trace_addrs = trace_addrs - m_trace_addrs
    assert aux_trace_addrs == tau_star

    lw_2 = batched_log_prob_sum(
        out, membership_filter({"x_2", "x_3"})
    ) + batched_log_prob_sum(out, membership_filter({"z_1"}))
    assert tensor_eq(_log_weight(out), lw_2), "_log_weight(p_out) is not expected lw_2"


def test_nested_marginal(simple2, simple4, simple5):
    s2, s4, s5 = primitive(simple2), primitive(simple4), primitive(simple5)

    out = extend(p=extend(p=s2, f=s4), f=s5)()
    assert _addrs(out) == {"x_2", "x_3", "z_2", "z_3", "z_1", "z_5"}

    p_nodes = list(filter(lambda kv: kv[0] not in {"z_1", "z_5"}, out.nodes.items()))
    assert len(p_nodes) > 0

    marg = get_marginal(out)[0]
    assert _addrs(marg) == {"x_2", "x_3", "z_2", "z_3"}


def test_compose_simple(simple1, simple3):
    s1, s3 = primitive(simple1), primitive(simple3)
    s1_out, s3_out = s1(), s3(None)
    replay_s1, replay_s3 = replay(s1, trace=s1_out), replay(s3, trace=s3_out)

    out = compose(q1=replay_s1, q2=replay_s3)()

    assert _addrs(out) == {"x_1", "x_2", "x_3", "z_1", "z_2", "z_3"}
    assert torch.equal(_log_weight(out), _log_weight(s1_out) + _log_weight(s3_out))

    assert torch.equal(
        _log_weight(out),
        sum(
            [
                batched_log_prob_sum(out, addr_filter(starts_with_x))
                for out in [s1_out, s3_out]
            ]
        ),
    )


def test_compose_with_plates(simple1, simple3):
    with pyro.plate("batch_dim", 3), pyro.plate("sample_dim", 7):
        s1, s3 = primitive(simple1), primitive(simple3)
        s1_out, s3_out = s1(), s3(None)
        replay_s1, replay_s3 = replay(s1, trace=s1_out), replay(s3, trace=s3_out)
        out = compose(q1=replay_s1, q2=replay_s3)()

    assert _addrs(out) == {"x_1", "x_2", "x_3", "z_1", "z_2", "z_3"}
    assert tensor_eq(_log_weight(out), _log_weight(s1_out) + _log_weight(s3_out))

    manual = sum(
        [
            batched_log_prob_sum(out, membership_filter({"x_1", "x_2", "x_3"}))
            for out in [s1_out, s3_out]
        ]
    )
    # type: ignore
    assert tensor_eq(_log_weight(out), manual)


@mark.skip()
def test_propose_(simple1, simple2):
    s1, s2 = primitive(simple1), primitive(simple2)
    s1_out, s2_out = s1(), s2()
    replay_s1, replay_s2 = replay(s1, trace=s1_out), replay(s2, trace=s2_out)

    q_out, p_out = s1_out, s2_out
    out = propose(p=replay_s2, q=replay_s1)()

    rho_q_addrs = {"x_1", "x_2", "z_1", "z_2"}
    tau_q_addrs = {"z_1", "z_2"}
    tau_p_addrs = {"z_2", "z_3"}
    nodes = rho_q_addrs - (tau_q_addrs - tau_p_addrs)

    assert _addrs(q_out) == {"z_1", "z_2", "x_1", "x_2"}
    assert _addrs(p_out) == {"z_2", "z_3", "x_2", "x_3"}

    # Compute stuff the same way as in the propose combinator to ensure numeric reproducability
    lu_1 = q_out.log_prob_sum(membership_filter(nodes))
    lu_star = torch.zeros(1)
    lw_1 = _log_weight(q_out)
    lv = _log_weight(p_out) - (lu_1 + lu_star)
    lw_out = lw_1 + lv

    assert torch.equal(lw_out.squeeze(), _log_weight(out)), "lw_out"


def test_propose(simple1, simple2, simple3, simple4):
    seed(7)

    s1, s3 = primitive(simple1), primitive(simple3)
    s1_out, s3_out = s1(), s3(None)
    replay_s1, replay_s3 = replay(s1, trace=s1_out), replay(s3, trace=s3_out)

    q = compose(q1=replay_s1, q2=replay_s3)
    q_out = q()

    tau_1 = {"z_1", "z_2", "z_3", "x_2", "x_3", "x_1"}
    assert _addrs(q_out) == tau_1
    assert torch.equal(_log_weight(q_out), _log_weight(s1_out) + _log_weight(s3_out))

    assert torch.equal(
        _log_weight(q_out),
        sum(
            [
                batched_log_prob_sum(out, addr_filter(starts_with_x))
                for out in [s1_out, s3_out]
            ]
        ),
    )

    lw_1 = q_out.log_prob_sum(membership_filter({"x_1", "x_2"})) + q_out.log_prob_sum(
        membership_filter({"x_3"})
    )
    assert lw_1 == _log_weight(q_out)

    s2, s4 = primitive(simple2), primitive(simple4)

    s2_out = s2()
    replay_s2 = replay(s2, trace=s2_out)
    s4_out = s4(valueat(s2_out, _RETURN))
    replay_s4 = replay(s4, trace=s4_out)

    p = with_substitution(extend(p=replay_s2, f=replay_s4), trace=q_out)
    # p = extend(p=replay_s2, f=replay_s4)
    p_out = p()
    # assert _addrs(p_out) == {"x_2", "x_3", "z_2", "z_3", "z_1"}
    # assert (
    #    _log_weight(p_out) == _log_weight(s2_out) + s4_out.log_prob_sum()
    # ), "target under substitution is wrong"
    #
    # p_nodes = list(
    #    filter(lambda kv: kv[0] in s2_out.nodes.keys(), p_out.nodes.items())
    # )
    # assert len(p_nodes) > 0
    # assert all(list(map(lambda kv: not is_auxiliary(kv[1]), p_nodes)))
    #
    # f_nodes = list(
    #    filter(lambda kv: kv[0] in s4_out.nodes.keys(), p_out.nodes.items())
    # )
    # assert len(f_nodes) > 0
    # assert all(list(map(lambda kv: is_auxiliary(kv[1]), f_nodes)))

    # Test extend inside propose
    tau_2 = {"z_2", "z_3", "x_2", "x_3"}
    trace_addrs = _addrs(p_out)
    m_trace = get_marginal(p_out)[0]
    m_trace_addrs = _addrs(m_trace)
    assert m_trace_addrs == tau_2

    tau_star = {"z_1"}
    aux_trace_addrs = trace_addrs - m_trace_addrs
    assert aux_trace_addrs == tau_star

    lw_2 = p_out.log_prob_sum(
        membership_filter({"z_2", "z_3", "x_2", "x_3"})
    ) + p_out.log_prob_sum(membership_filter({"z_1"}))
    assert lw_2 == _log_weight(
        p_out
    ), f"_log_weight(p_out) ({_log_weight(p_out)}) is not expected lw_2 ({lw_2})"

    out = propose(p=p, q=q)()

    # Compute weight the same way as inside the propose combinator for reproduceability
    lu_1 = q_out.log_prob_sum(membership_filter({"x_2", "x_3", "x_1", "z_2", "z_3"}))
    lu_star = p_out.log_prob_sum(membership_filter({"z_1"}))
    lv = lw_2 - (lu_1 + lu_star)
    lw_out = lw_1 + lv
    #   (x_1 x_2 x_3) * ((z_2 z_3  *  x_2 x_3) * (z_1))
    #  ------------------------------------------------
    #  ((x_1 x_2 x_3  *   z_2 z_3)             * (z_1))
    assert torch.isclose(
        lw_out, _log_weight(out)
    ), "final weight, can be a bit off if addition happens out-of-order"


def test_propose_with_plates(simple1, simple2, simple3, simple4):
    seed(7)
    with pyro.plate("sample", 7), pyro.plate("batch", 3):
        s1, s3 = primitive(simple1), primitive(simple3)
        s1_out, s3_out = s1(), s3(None)
        replay_s1, replay_s3 = replay(s1, trace=s1_out), replay(s3, trace=s3_out)

        q = compose(q1=replay_s1, q2=replay_s3)
        q_out = q()

        tau_1 = {"z_1", "z_2", "z_3", "x_2", "x_3", "x_1"}
        assert _addrs(q_out) == tau_1
        assert tensor_eq(
            _log_weight(q_out), _log_weight(s1_out) + _log_weight(s3_out)
        ), "we can manually reconstruct log weight from the previous output weights"

        assert tensor_eq(
            _log_weight(q_out),
            sum(
                [
                    batched_log_prob_sum(out, addr_filter(starts_with_x))
                    for out in [s1_out, s3_out]
                ]
            ),
        ), "we can manually reconstruct log weight from original traces"

        # NOTE: order of addition matters for precise floating point math
        lw_1 = batched_log_prob_sum(
            q_out, membership_filter({"x_1", "x_2"})
        ) + batched_log_prob_sum(q_out, membership_filter({"x_3"}))
        assert tensor_eq(
            lw_1, _log_weight(q_out)
        ), "we can manually reconstruct log weight from the output trace"

        s2, s4 = primitive(simple2), primitive(simple4)

        s2_out = s2()
        replay_s2 = replay(s2, trace=s2_out)
        s4_out = s4(valueat(s2_out, _RETURN))
        replay_s4 = replay(s4, trace=s4_out)

        p = with_substitution(extend(p=replay_s2, f=replay_s4), trace=q_out)
        p_out = p()

        # Test extend inside propose
        tau_2 = {"z_2", "z_3", "x_2", "x_3"}
        trace_addrs = _addrs(p_out)
        m_trace = get_marginal(p_out)[0]
        m_trace_addrs = _addrs(m_trace)
        assert m_trace_addrs == tau_2

        tau_star = {"z_1"}
        aux_trace_addrs = trace_addrs - m_trace_addrs
        assert aux_trace_addrs == tau_star

        lw_2 = batched_log_prob_sum(
            p_out, membership_filter({"z_2", "z_3", "x_2", "x_3"})
        ) + batched_log_prob_sum(p_out, membership_filter({"z_1"}))
        assert tensor_eq(
            lw_2, _log_weight(p_out)
        ), f"_log_weight(p_out) ({_log_weight(p_out)}) is not expected lw_2 ({lw_2})"

        out = propose(p=p, q=q)()

        # Compute weight the same way as inside the propose combinator for reproduceability
        lu_1 = batched_log_prob_sum(
            q_out, membership_filter({"x_2", "x_3", "x_1", "z_2", "z_3"})
        )
        lu_star = batched_log_prob_sum(p_out, membership_filter({"z_1"}))
        lv = lw_2 - (lu_1 + lu_star)
        lw_out = lw_1 + lv
        assert torch.allclose(lw_out, _log_weight(out)), (
            "final weight, can be a bit off when addition happens out-of-order but computed:\n"
            + f"{lw_out}\nvs output:\n{_log_weight(out)}"
        )


def test_propose_output(simple1, simple2, simple3, simple4):
    seed(7)
    with pyro.plate("sample", 7), pyro.plate("batch", 3):
        s1, s3 = primitive(simple1), primitive(simple3)
        s1_out, s3_out = s1(), s3(None)
        replay_s1, replay_s3 = replay(s1, trace=s1_out), replay(s3, trace=s3_out)

        q = compose(q1=replay_s1, q2=replay_s3)
        q_out = q()
        assert q_out.nodes["_RETURN"]["value"] is None

        s2, s4 = primitive(simple2), primitive(simple4)

        s2_out = s2()
        replay_s2 = replay(s2, trace=s2_out)
        s4_out = s4(valueat(s2_out, _RETURN))
        replay_s4 = replay(s4, trace=s4_out)

        p = with_substitution(extend(p=replay_s2, f=replay_s4), trace=q_out)
        p_out = p()
        assert p_out.nodes["_RETURN"]["value"] == tensor_of(2)

        m_trace, m_output = get_marginal(p_out)
        assert m_output["value"] == tensor_of(1)
