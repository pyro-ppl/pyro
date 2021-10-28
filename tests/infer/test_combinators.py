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
    Out,
    addr_filter,
    compose,
    extend,
    get_marginal,
    is_auxiliary,
    membership_filter,
    not_auxiliary,
    primitive,
    propose,
    with_substitution,
)
from pyro.poutine import Trace, replay


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
    def model():
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
    tr0, tr1 = primitive_model().trace, non_overlapping_primitive().trace
    tr0_names = {name for name, _ in tr0.nodes.items()}
    for name, _ in tr1.nodes.items():
        assert name not in tr0_names, f"{name} is in both traces!"


def assert_log_weight_zero(primitive_output):
    lw = primitive_output.log_weight
    assert isinstance(lw, Tensor) and lw == 0.0


# primitive expression tests
# ===========================================================================


def test_constructor(model0):
    m = primitive(model0)
    assert_no_observe(m)
    out = m()
    assert_no_observe(m)
    assert_log_weight_zero(out)

    o = out.output[0]
    assert isinstance(o, str) and o in {"cloudy", "sunny"}
    assert isinstance(out.trace, Trace)


def test_no_overlapping_variables(model0, model1):
    m0 = primitive(model0)
    m1 = primitive(model1)
    assert_no_observe(m0)
    assert_no_observe(m1)
    assert_no_overlap(m0, m1)


def test_with_substitution(model0):
    p = primitive(model0)
    q = primitive(model0)
    p_out = p()
    q_out = with_substitution(q, trace=p_out.trace)()

    p_addrs = set(p_out.trace.nodes.keys())
    q_addrs = set(q_out.trace.nodes.keys())
    assert p_addrs.intersection(q_addrs) == p_addrs.union(q_addrs)

    def valueat(o, a):
        return o.trace.nodes[a]["value"]

    for a in p_addrs:
        assert q_out.trace.nodes[a]["value"] == p_out.trace.nodes[a]["value"]
        assert valueat(q_out, a) == valueat(p_out, a)
        assert q_out.trace.nodes[a]["infer"]["substituted"] is True

    assert p_out.output == q_out.output
    assert q_out.log_weight != 0.0


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

    assert p_out.log_weight.shape == torch.Size(
        [7, 5, 3]
    ), "plated calls should return weights for independent dimensions"

    with pyro.plate("data", 3):
        q = primitive(model)
        q_out = q(inp)

        p = primitive(model)
        p_out = with_substitution(p, trace=q_out.trace)(inp)

    p_addrs = set(p_out.trace.nodes.keys())
    q_addrs = set(q_out.trace.nodes.keys())
    assert p_addrs.intersection(q_addrs) == p_addrs.union(q_addrs)

    for a in p_addrs:
        pnode = p_out.trace.nodes[a]
        qnode = q_out.trace.nodes[a]
        assert torch.equal(pnode["value"], qnode["value"])
        assert torch.equal(pnode["value"], qnode["value"])
        assert pnode["is_observed"] ^ pnode["infer"].get("substituted", False)

    assert p_out.output == q_out.output
    assert torch.all(
        torch.ne(q_out.log_weight, torch.zeros(q_out.log_weight.shape))
    ).item()


def starts_with_x(n):
    return n[0] == "x"


# test simple programs
def test_simple1(simple1):
    s1_out = primitive(simple1)()
    assert set(s1_out.trace.nodes.keys()) == {"z_1", "z_2", "x_1", "x_2"}
    assert s1_out.log_weight == s1_out.trace.log_prob_sum(addr_filter(starts_with_x))


def test_simple2(simple2):
    out = primitive(simple2)()
    assert set(out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3"}
    assert out.log_weight == out.trace.log_prob_sum(addr_filter(starts_with_x))
    assert out.log_weight != 0.0


def test_simple3(simple3):
    out = primitive(simple3)()
    assert set(out.trace.nodes.keys()) == {"z_3", "x_3"}
    assert out.log_weight == out.trace.log_prob_sum(addr_filter(starts_with_x))


def test_simple4(simple4):
    out = primitive(simple4)(tensor([1]))
    assert set(out.trace.nodes.keys()) == {"z_1"}
    assert out.log_weight == 0.0


def test_simple_substitution(simple1, simple2):
    s1, s2 = primitive(simple1), primitive(simple2)
    s1_out = s1()
    s2_out = with_substitution(s2, trace=s1_out.trace)()

    rho_f_addrs = {"x_2", "x_3", "z_2", "z_3"}
    tau_f_addrs = {"z_2", "z_3"}
    tau_p_addrs = {"z_1", "z_2"}
    nodes = rho_f_addrs - (tau_f_addrs - tau_p_addrs)
    lw_out = s2_out.trace.log_prob_sum(membership_filter(nodes))

    assert (lw_out == s2_out.log_weight).all()
    assert len(set({"z_1", "x_1"}).intersection(set(s2_out.trace.nodes.keys()))) == 0


def test_extend_unconditioned_no_plates(simple2, simple4):
    s2, s4 = primitive(simple2), primitive(simple4)

    p_out = s2()
    replay_s2 = poutine.replay(s2, trace=p_out.trace)
    tau_2 = {"z_2", "z_3", "x_2", "x_3"}
    assert set(p_out.trace.nodes.keys()) == tau_2
    assert p_out.log_weight == batched_log_prob_sum(p_out, addr_filter(starts_with_x))

    f_out = s4(p_out.output)
    replay_s4 = poutine.replay(s4, trace=f_out.trace)
    tau_star = {"z_1"}
    assert set(f_out.trace.nodes.keys()) == tau_star
    assert f_out.log_weight == 0.0

    out = extend(p=replay_s2, f=replay_s4)()
    assert set(out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3", "z_1"}
    assert out.log_weight == p_out.log_weight + f_out.trace.log_prob_sum()

    def in_keys(out):
        def go(kv):
            return kv[0] in out.trace.nodes.keys()

        return go

    p_nodes = list(filter(in_keys(p_out), out.trace.nodes.items()))
    assert len(p_nodes) > 0
    assert all(list(map(not_auxiliary, map(itemgetter(1), p_nodes))))

    f_nodes = list(
        filter(lambda kv: kv[0] in f_out.trace.nodes.keys(), out.trace.nodes.items())
    )
    assert len(f_nodes) > 0
    assert all(list(map(is_auxiliary, map(itemgetter(1), f_nodes))))

    # Test extend log_weight
    trace_addrs = set(out.trace.nodes.keys())
    m_trace_addrs = set(get_marginal(out.trace).nodes.keys())
    assert m_trace_addrs == tau_2

    aux_trace_addrs = trace_addrs - m_trace_addrs
    assert aux_trace_addrs == tau_star

    lw_2 = out.trace.log_prob_sum(
        membership_filter({"x_2", "x_3"})
    ) + out.trace.log_prob_sum(membership_filter({"z_1"}))
    assert lw_2 == out.log_weight, "p_out.log_weight is not expected lw_2"


def tensor_eq(t0: Tensor, t1: Tensor) -> bool:
    if t0.shape != t1.shape:
        raise AssertionError(f"shape mismatch: {t0.shape} vs {t1.shape}")
    return bool(torch.all(torch.eq(t0, t1)).item())


def batched_log_prob_sum(
    o: Out, site_filter: Callable[[str, Node], bool] = (lambda a, b: True)
) -> Tensor:
    o.trace.compute_log_prob(site_filter=site_filter)
    lp = torch.stack(
        [n["log_prob"] for k, n in o.trace.nodes.items() if site_filter(k, n)]
    ).sum(dim=0)
    return lp if len(lp.shape) > 0 else lp.unsqueeze(0)


def test_extend_unconditioned_with_plates(simple2, simple4):
    s2, s4 = primitive(simple2), primitive(simple4)

    with pyro.plate("batch", 3), pyro.plate("samples", 5):
        p_out = s2()
        replay_s2 = poutine.replay(s2, trace=p_out.trace)
        f_out = s4(p_out.output)
        replay_s4 = poutine.replay(s4, trace=f_out.trace)

    tau_2 = {"z_2", "z_3", "x_2", "x_3"}
    assert set(p_out.trace.nodes.keys()) == tau_2
    assert tensor_eq(
        p_out.log_weight, batched_log_prob_sum(p_out, lambda n, _: n[0] == "x")
    )
    tau_star = {"z_1"}
    assert set(f_out.trace.nodes.keys()) == tau_star
    assert (
        f_out.log_weight.shape == torch.Size([1]) and f_out.log_weight == 0.0
    ), "kernel's log weight is scalar of zero"

    with pyro.plate("batch", 3), pyro.plate("samples", 5):
        out = extend(p=replay_s2, f=replay_s4)()

    assert set(out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3", "z_1"}

    assert tensor_eq(out.log_weight, p_out.log_weight + batched_log_prob_sum(f_out))

    p_nodes = list(
        filter(lambda kv: kv[0] in p_out.trace.nodes.keys(), out.trace.nodes.items())
    )
    assert len(p_nodes) > 0
    assert all(list(map(lambda kv: not is_auxiliary(kv[1]), p_nodes)))

    f_nodes = list(
        filter(lambda kv: kv[0] in f_out.trace.nodes.keys(), out.trace.nodes.items())
    )
    assert len(f_nodes) > 0
    assert all(list(map(lambda kv: is_auxiliary(kv[1]), f_nodes)))

    # Test extend log_weight
    trace_addrs = set(out.trace.nodes.keys())
    m_trace_addrs = set(get_marginal(out.trace).nodes.keys())
    assert m_trace_addrs == tau_2

    aux_trace_addrs = trace_addrs - m_trace_addrs
    assert aux_trace_addrs == tau_star

    lw_2 = batched_log_prob_sum(
        out, membership_filter({"x_2", "x_3"})
    ) + batched_log_prob_sum(out, membership_filter({"z_1"}))
    assert tensor_eq(out.log_weight, lw_2), "p_out.log_weight is not expected lw_2"


def test_nested_marginal(simple2, simple4, simple5):
    s2, s4, s5 = primitive(simple2), primitive(simple4), primitive(simple5)

    out = extend(p=extend(p=s2, f=s4), f=s5)()
    assert set(out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3", "z_1", "z_5"}

    p_nodes = list(
        filter(lambda kv: kv[0] not in {"z_1", "z_5"}, out.trace.nodes.items())
    )
    assert len(p_nodes) > 0

    marg = get_marginal(out.trace)
    assert set(marg.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3"}


def test_compose(simple1, simple3):
    s1, s3 = primitive(simple1), primitive(simple3)
    s1_out, s3_out = s1(), s3()
    replay_s1, replay_s3 = replay(s1, trace=s1_out.trace), replay(
        s3, trace=s3_out.trace
    )

    out = compose(q1=replay_s1, q2=replay_s3)()

    assert set(out.trace.nodes.keys()) == {"x_1", "x_2", "x_3", "z_1", "z_2", "z_3"}
    assert torch.equal(out.log_weight, s1_out.log_weight + s3_out.log_weight)

    assert torch.equal(
        out.log_weight,
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
        s1_out, s3_out = s1(), s3()
        replay_s1, replay_s3 = replay(s1, trace=s1_out.trace), replay(
            s3, trace=s3_out.trace
        )
        out = compose(q1=replay_s1, q2=replay_s3)()

    assert set(out.trace.nodes.keys()) == {"x_1", "x_2", "x_3", "z_1", "z_2", "z_3"}
    assert tensor_eq(out.log_weight, s1_out.log_weight + s3_out.log_weight)

    manual = sum(
        [
            batched_log_prob_sum(out, membership_filter({"x_1", "x_2", "x_3"}))
            for out in [s1_out, s3_out]
        ]
    )
    # type: ignore
    assert tensor_eq(out.log_weight, manual)


@mark.skip()
def test_propose_(simple1, simple2):
    s1, s2 = primitive(simple1), primitive(simple2)
    s1_out, s2_out = s1(), s2()
    replay_s1, replay_s2 = replay(s1, trace=s1_out.trace), replay(
        s2, trace=s2_out.trace
    )

    q_out, p_out = s1_out, s2_out
    out = propose(p=replay_s2, q=replay_s1)()

    rho_q_addrs = {"x_1", "x_2", "z_1", "z_2"}
    tau_q_addrs = {"z_1", "z_2"}
    tau_p_addrs = {"z_2", "z_3"}
    nodes = rho_q_addrs - (tau_q_addrs - tau_p_addrs)

    assert set(q_out.trace.nodes.keys()) == {"z_1", "z_2", "x_1", "x_2"}
    assert set(p_out.trace.nodes.keys()) == {"z_2", "z_3", "x_2", "x_3"}

    # Compute stuff the same way as in the propose combinator to ensure numeric reproducability
    lu_1 = q_out.trace.log_prob_sum(membership_filter(nodes))
    lu_star = torch.zeros(1)
    lw_1 = q_out.log_weight
    lv = p_out.log_weight - (lu_1 + lu_star)
    lw_out = lw_1 + lv

    assert torch.equal(lw_out.squeeze(), out.log_weight), "lw_out"


def test_propose(simple1, simple2, simple3, simple4):
    seed(7)

    s1, s3 = primitive(simple1), primitive(simple3)
    s1_out, s3_out = s1(), s3()
    replay_s1, replay_s3 = replay(s1, trace=s1_out.trace), replay(
        s3, trace=s3_out.trace
    )

    q = compose(q1=replay_s1, q2=replay_s3)
    q_out = q()

    tau_1 = {"z_1", "z_2", "z_3", "x_2", "x_3", "x_1"}
    assert set(q_out.trace.nodes.keys()) == tau_1
    assert torch.equal(q_out.log_weight, s1_out.log_weight + s3_out.log_weight)

    assert torch.equal(
        q_out.log_weight,
        sum(
            [
                batched_log_prob_sum(out, addr_filter(starts_with_x))
                for out in [s1_out, s3_out]
            ]
        ),
    )

    lw_1 = q_out.trace.log_prob_sum(
        membership_filter({"x_1", "x_2"})
    ) + q_out.trace.log_prob_sum(membership_filter({"x_3"}))
    assert lw_1 == q_out.log_weight

    s2, s4 = primitive(simple2), primitive(simple4)

    s2_out = s2()
    replay_s2 = replay(s2, trace=s2_out.trace)
    s4_out = s4(s2_out.output)
    replay_s4 = replay(s4, trace=s4_out.trace)

    p = with_substitution(extend(p=replay_s2, f=replay_s4), trace=q_out.trace)
    # p = extend(p=replay_s2, f=replay_s4)
    p_out = p()
    # assert set(p_out.trace.nodes.keys()) == {"x_2", "x_3", "z_2", "z_3", "z_1"}
    # assert (
    #    p_out.log_weight == s2_out.log_weight + s4_out.trace.log_prob_sum()
    # ), "target under substitution is wrong"
    #
    # p_nodes = list(
    #    filter(lambda kv: kv[0] in s2_out.trace.nodes.keys(), p_out.trace.nodes.items())
    # )
    # assert len(p_nodes) > 0
    # assert all(list(map(lambda kv: not is_auxiliary(kv[1]), p_nodes)))
    #
    # f_nodes = list(
    #    filter(lambda kv: kv[0] in s4_out.trace.nodes.keys(), p_out.trace.nodes.items())
    # )
    # assert len(f_nodes) > 0
    # assert all(list(map(lambda kv: is_auxiliary(kv[1]), f_nodes)))

    # Test extend inside propose
    tau_2 = {"z_2", "z_3", "x_2", "x_3"}
    trace_addrs = set(p_out.trace.nodes.keys())
    m_trace = get_marginal(p_out.trace)
    m_trace_addrs = set(m_trace.nodes.keys())
    assert m_trace_addrs == tau_2

    tau_star = {"z_1"}
    aux_trace_addrs = trace_addrs - m_trace_addrs
    assert aux_trace_addrs == tau_star

    lw_2 = p_out.trace.log_prob_sum(
        membership_filter({"z_2", "z_3", "x_2", "x_3"})
    ) + p_out.trace.log_prob_sum(membership_filter({"z_1"}))
    assert (
        lw_2 == p_out.log_weight
    ), f"p_out.log_weight ({p_out.log_weight}) is not expected lw_2 ({lw_2})"

    out = propose(p=p, q=q)()

    # Compute weight the same way as inside the propose combinator for reproduceability
    lu_1 = q_out.trace.log_prob_sum(
        membership_filter({"x_2", "x_3", "x_1", "z_2", "z_3"})
    )
    lu_star = p_out.trace.log_prob_sum(membership_filter({"z_1"}))
    lv = lw_2 - (lu_1 + lu_star)
    lw_out = lw_1 + lv
    #   (x_1 x_2 x_3) * ((z_2 z_3  *  x_2 x_3) * (z_1))
    #  ------------------------------------------------
    #  ((x_1 x_2 x_3  *   z_2 z_3)             * (z_1))
    assert torch.isclose(
        lw_out, out.log_weight
    ), "final weight, can be a bit off if addition happens out-of-order"


def test_propose_with_plates(simple1, simple2, simple3, simple4):
    seed(7)
    with pyro.plate("sample", 7), pyro.plate("batch", 3):
        s1, s3 = primitive(simple1), primitive(simple3)
        s1_out, s3_out = s1(), s3()
        replay_s1, replay_s3 = replay(s1, trace=s1_out.trace), replay(
            s3, trace=s3_out.trace
        )

        q = compose(q1=replay_s1, q2=replay_s3)
        q_out = q()

        tau_1 = {"z_1", "z_2", "z_3", "x_2", "x_3", "x_1"}
        assert set(q_out.trace.nodes.keys()) == tau_1
        assert tensor_eq(
            q_out.log_weight, s1_out.log_weight + s3_out.log_weight
        ), "we can manually reconstruct log weight from the previous output weights"

        assert tensor_eq(
            q_out.log_weight,
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
            lw_1, q_out.log_weight
        ), "we can manually reconstruct log weight from the output trace"

        s2, s4 = primitive(simple2), primitive(simple4)

        s2_out = s2()
        replay_s2 = replay(s2, trace=s2_out.trace)
        s4_out = s4(s2_out.output)
        replay_s4 = replay(s4, trace=s4_out.trace)

        p = with_substitution(extend(p=replay_s2, f=replay_s4), trace=q_out.trace)
        p_out = p()

        # Test extend inside propose
        tau_2 = {"z_2", "z_3", "x_2", "x_3"}
        trace_addrs = set(p_out.trace.nodes.keys())
        m_trace = get_marginal(p_out.trace)
        m_trace_addrs = set(m_trace.nodes.keys())
        assert m_trace_addrs == tau_2

        tau_star = {"z_1"}
        aux_trace_addrs = trace_addrs - m_trace_addrs
        assert aux_trace_addrs == tau_star

        lw_2 = batched_log_prob_sum(
            p_out, membership_filter({"z_2", "z_3", "x_2", "x_3"})
        ) + batched_log_prob_sum(p_out, membership_filter({"z_1"}))
        assert tensor_eq(
            lw_2, p_out.log_weight
        ), f"p_out.log_weight ({p_out.log_weight}) is not expected lw_2 ({lw_2})"

        out = propose(p=p, q=q)()

        # Compute weight the same way as inside the propose combinator for reproduceability
        lu_1 = batched_log_prob_sum(
            q_out, membership_filter({"x_2", "x_3", "x_1", "z_2", "z_3"})
        )
        lu_star = batched_log_prob_sum(p_out, membership_filter({"z_1"}))
        lv = lw_2 - (lu_1 + lu_star)
        lw_out = lw_1 + lv
        assert torch.allclose(lw_out, out.log_weight), (
            "final weight, can be a bit off when addition happens out-of-order but computed:\n"
            + f"{lw_out}\nvs output:\n{out.log_weight}"
        )
