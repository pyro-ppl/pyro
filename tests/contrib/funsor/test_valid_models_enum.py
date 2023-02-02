# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import os
from collections import defaultdict
from queue import LifoQueue

import pytest
import torch

from pyro.infer.enum import iter_discrete_escape, iter_discrete_extend
from pyro.ops.indexing import Vindex
from pyro.poutine import Trace
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_traceenum_requirements

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor
    from pyro.contrib.funsor.handlers.runtime import _DIM_STACK

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import handlers, infer, pyro, pyro_backend
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)

# default to 2, which checks that packed but not unpacked shapes match
_NAMED_TEST_STRENGTH = int(os.environ.get("NAMED_TEST_STRENGTH", 2))


def assert_ok(model, guide=None, max_plate_nesting=None, **kwargs):
    """
    Assert that enumeration runs...
    """
    with pyro_backend("pyro"):
        pyro.clear_param_store()

    if guide is None:
        guide = lambda **kwargs: None  # noqa: E731

    q_pyro, q_funsor = LifoQueue(), LifoQueue()
    q_pyro.put(Trace())
    q_funsor.put(Trace())

    while not q_pyro.empty() and not q_funsor.empty():
        with pyro_backend("pyro"):
            with handlers.enum(first_available_dim=-max_plate_nesting - 1):
                guide_tr_pyro = handlers.trace(
                    handlers.queue(
                        guide,
                        q_pyro,
                        escape_fn=iter_discrete_escape,
                        extend_fn=iter_discrete_extend,
                    )
                ).get_trace(**kwargs)
                tr_pyro = handlers.trace(
                    handlers.replay(model, trace=guide_tr_pyro)
                ).get_trace(**kwargs)

        with pyro_backend("contrib.funsor"):
            with handlers.enum(first_available_dim=-max_plate_nesting - 1):
                guide_tr_funsor = handlers.trace(
                    handlers.queue(
                        guide,
                        q_funsor,
                        escape_fn=iter_discrete_escape,
                        extend_fn=iter_discrete_extend,
                    )
                ).get_trace(**kwargs)
                tr_funsor = handlers.trace(
                    handlers.replay(model, trace=guide_tr_funsor)
                ).get_trace(**kwargs)

        # make sure all dimensions were cleaned up
        assert _DIM_STACK.local_frame is _DIM_STACK.global_frame
        assert (
            not _DIM_STACK.global_frame.name_to_dim
            and not _DIM_STACK.global_frame.dim_to_name
        )
        assert _DIM_STACK.outermost is None

        tr_pyro = prune_subsample_sites(tr_pyro.copy())
        tr_funsor = prune_subsample_sites(tr_funsor.copy())
        _check_traces(tr_pyro, tr_funsor)


def _check_traces(tr_pyro, tr_funsor):
    assert tr_pyro.nodes.keys() == tr_funsor.nodes.keys()
    tr_pyro.compute_log_prob()
    tr_funsor.compute_log_prob()
    tr_pyro.pack_tensors()

    symbol_to_name = {
        node["infer"]["_enumerate_symbol"]: name
        for name, node in tr_pyro.nodes.items()
        if node["type"] == "sample"
        and not node["is_observed"]
        and node["infer"].get("enumerate") == "parallel"
    }
    symbol_to_name.update(
        {symbol: name for name, symbol in tr_pyro.plate_to_symbol.items()}
    )

    if _NAMED_TEST_STRENGTH >= 1:
        # coarser check: enumeration requirements satisfied
        check_traceenum_requirements(tr_pyro, Trace())
        check_traceenum_requirements(tr_funsor, Trace())
        try:
            # coarser check: number of elements and squeezed shapes
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node["type"] != "sample":
                    continue
                funsor_node = tr_funsor.nodes[name]
                assert (
                    pyro_node["packed"]["log_prob"].numel()
                    == funsor_node["log_prob"].numel()
                )
                assert (
                    pyro_node["packed"]["log_prob"].shape
                    == funsor_node["log_prob"].squeeze().shape
                )
                assert frozenset(
                    f for f in pyro_node["cond_indep_stack"] if f.vectorized
                ) == frozenset(
                    f for f in funsor_node["cond_indep_stack"] if f.vectorized
                )
        except AssertionError:
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node["type"] != "sample":
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_packed_shape = pyro_node["packed"]["log_prob"].shape
                funsor_packed_shape = funsor_node["log_prob"].squeeze().shape
                if pyro_packed_shape != funsor_packed_shape:
                    err_str = "==> (dep mismatch) {}".format(name)
                else:
                    err_str = name
                print(
                    err_str,
                    "Pyro: {} vs Funsor: {}".format(
                        pyro_packed_shape, funsor_packed_shape
                    ),
                )
            raise

    if _NAMED_TEST_STRENGTH >= 2:
        try:
            # medium check: unordered packed shapes match
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node["type"] != "sample":
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_names = frozenset(
                    symbol_to_name[d]
                    for d in pyro_node["packed"]["log_prob"]._pyro_dims
                )
                funsor_names = frozenset(funsor_node["funsor"]["log_prob"].inputs)
                assert pyro_names == frozenset(
                    name.replace("__PARTICLES", "") for name in funsor_names
                )
        except AssertionError:
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node["type"] != "sample":
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_names = frozenset(
                    symbol_to_name[d]
                    for d in pyro_node["packed"]["log_prob"]._pyro_dims
                )
                funsor_names = frozenset(funsor_node["funsor"]["log_prob"].inputs)
                if pyro_names != funsor_names:
                    err_str = "==> (packed mismatch) {}".format(name)
                else:
                    err_str = name
                print(
                    err_str,
                    "Pyro: {} vs Funsor: {}".format(
                        sorted(tuple(pyro_names)), sorted(tuple(funsor_names))
                    ),
                )
            raise

    if _NAMED_TEST_STRENGTH >= 3:
        try:
            # finer check: exact match with unpacked Pyro shapes
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node["type"] != "sample":
                    continue
                funsor_node = tr_funsor.nodes[name]
                assert pyro_node["log_prob"].shape == funsor_node["log_prob"].shape
                assert pyro_node["value"].shape == funsor_node["value"].shape
        except AssertionError:
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node["type"] != "sample":
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_shape = pyro_node["log_prob"].shape
                funsor_shape = funsor_node["log_prob"].shape
                if pyro_shape != funsor_shape:
                    err_str = "==> (unpacked mismatch) {}".format(name)
                else:
                    err_str = name
                print(
                    err_str, "Pyro: {} vs Funsor: {}".format(pyro_shape, funsor_shape)
                )
            raise


@pytest.mark.parametrize("history", [1, 2, 3])
def test_enum_recycling_chain_iter(history):
    @infer.config_enumerate
    def model():
        p = torch.tensor([[0.2, 0.8], [0.1, 0.9]])

        xs = [0]
        for t in pyro.markov(range(100), history=history):
            xs.append(pyro.sample("x_{}".format(t), dist.Categorical(p[xs[-1]])))
        assert all(x.dim() <= history + 1 for x in xs[1:])

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize("history", [2, 3])
def test_enum_recycling_chain_iter_interleave_parallel_sequential(history):
    def model():
        p = torch.tensor([[0.2, 0.8], [0.1, 0.9]])

        xs = [0]
        for t in pyro.markov(range(10), history=history):
            xs.append(
                pyro.sample(
                    "x_{}".format(t),
                    dist.Categorical(p[xs[-1]]),
                    infer={"enumerate": ("sequential", "parallel")[t % 2]},
                )
            )
        assert all(x.dim() <= history + 1 for x in xs[1:])

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize("history", [1, 2, 3])
def test_enum_recycling_chain_while(history):
    @infer.config_enumerate
    def model():
        p = torch.tensor([[0.2, 0.8], [0.1, 0.9]])

        xs = [0]
        c = pyro.markov(history=history)
        with contextlib.ExitStack() as stack:
            for t in range(100):
                stack.enter_context(c)
                xs.append(pyro.sample("x_{}".format(t), dist.Categorical(p[xs[-1]])))
            assert all(x.dim() <= history + 1 for x in xs[1:])

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize("history", [1, 2, 3])
def test_enum_recycling_chain_recur(history):
    @infer.config_enumerate
    def model():
        p = torch.tensor([[0.2, 0.8], [0.1, 0.9]])

        x = 0

        @pyro.markov(history=history)
        def fn(t, x):
            x = pyro.sample("x_{}".format(t), dist.Categorical(p[x]))
            assert x.dim() <= history + 1
            return x if t >= 100 else fn(t + 1, x)

        return fn(0, x)

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize("use_vindex", [False, True])
@pytest.mark.parametrize("markov", [False, True])
def test_enum_recycling_dbn(markov, use_vindex):
    #    x --> x --> x  enum "state"
    # y  |  y  |  y  |  enum "occlusion"
    #  \ |   \ |   \ |
    #    z     z     z  obs

    @infer.config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        q = pyro.param("q", torch.ones(2))
        r = pyro.param("r", torch.ones(3, 2, 4))

        x = 0
        times = pyro.markov(range(100)) if markov else range(11)
        for t in times:
            x = pyro.sample("x_{}".format(t), dist.Categorical(p[x]))
            y = pyro.sample("y_{}".format(t), dist.Categorical(q))
            if use_vindex:
                probs = Vindex(r)[x, y]
            else:
                z_ind = torch.arange(4, dtype=torch.long)
                probs = r[x.unsqueeze(-1), y.unsqueeze(-1), z_ind]
            pyro.sample(
                "z_{}".format(t), dist.Categorical(probs), obs=torch.tensor(0.0)
            )

    assert_ok(model, max_plate_nesting=0)


def test_enum_recycling_nested():
    # (x)
    #   \
    #    y0---(y1)--(y2)
    #    |     |     |
    #   z00   z10   z20
    #    |     |     |
    #   z01   z11  (z21)
    #    |     |     |
    #   z02   z12   z22 <-- what can this depend on?
    #
    # markov dependencies
    # -------------------
    #   x:
    #  y0: x
    # z00: x y0
    # z01: x y0 z00
    # z02: x y0 z01
    #  y1: x y0
    # z10: x y0 y1
    # z11: x y0 y1 z10
    # z12: x y0 y1 z11
    #  y2: x y1
    # z20: x y1 y2
    # z21: x y1 y2 z20
    # z22: x y1 y2 z21

    @infer.config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        x = pyro.sample("x", dist.Categorical(p[0]))
        y = x
        for i in pyro.markov(range(10)):
            y = pyro.sample("y_{}".format(i), dist.Categorical(p[y]))
            z = y
            for j in pyro.markov(range(10)):
                z = pyro.sample("z_{}_{}".format(i, j), dist.Categorical(p[z]))

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.xfail(reason="Pyro behavior here appears to be incorrect")
@pytest.mark.parametrize("grid_size", [4, 20])
@pytest.mark.parametrize("use_vindex", [False, True])
def test_enum_recycling_grid(grid_size, use_vindex):
    #  x---x---x---x    -----> i
    #  |   |   |   |   |
    #  x---x---x---x   |
    #  |   |   |   |   V
    #  x---x---x--(x)  j
    #  |   |   |   |
    #  x---x--(x)--x <-- what can this depend on?

    @infer.config_enumerate
    def model():
        p = pyro.param("p_leaf", torch.ones(2, 2, 2))
        x = defaultdict(lambda: torch.tensor(0))
        y_axis = pyro.markov(range(grid_size), keep=True)
        for i in pyro.markov(range(grid_size)):
            for j in y_axis:
                if use_vindex:
                    probs = Vindex(p)[x[i - 1, j], x[i, j - 1]]
                else:
                    ind = torch.arange(2, dtype=torch.long)
                    probs = p[x[i - 1, j].unsqueeze(-1), x[i, j - 1].unsqueeze(-1), ind]
                x[i, j] = pyro.sample("x_{}_{}".format(i, j), dist.Categorical(probs))

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize("max_plate_nesting", [0, 1, 2])
@pytest.mark.parametrize("depth", [3, 5, 7])
@pytest.mark.parametrize("history", [1, 2])
def test_enum_recycling_reentrant_history(max_plate_nesting, depth, history):
    data = (True, False)
    for i in range(depth):
        data = (data, data, False)

    def model_(**kwargs):
        @pyro.markov(history=history)
        def model(data, state=0, address=""):
            if isinstance(data, bool):
                p = pyro.param("p_leaf", torch.ones(10))
                pyro.sample(
                    "leaf_{}".format(address),
                    dist.Bernoulli(p[state]),
                    obs=torch.tensor(1.0 if data else 0.0),
                )
            else:
                assert isinstance(data, tuple)
                p = pyro.param("p_branch", torch.ones(10, 10))
                for branch, letter in zip(data, "abcdefg"):
                    next_state = pyro.sample(
                        "branch_{}".format(address + letter),
                        dist.Categorical(p[state]),
                        infer={"enumerate": "parallel"},
                    )
                    model(branch, next_state, address + letter)

        return model(**kwargs)

    assert_ok(model_, max_plate_nesting=max_plate_nesting, data=data)


@pytest.mark.parametrize("max_plate_nesting", [0, 1, 2])
@pytest.mark.parametrize("depth", [3, 5, 7])
def test_enum_recycling_mutual_recursion(max_plate_nesting, depth):
    data = (True, False)
    for i in range(depth):
        data = (data, data, False)

    def model_(**kwargs):
        def model_leaf(data, state=0, address=""):
            p = pyro.param("p_leaf", torch.ones(10))
            pyro.sample(
                "leaf_{}".format(address),
                dist.Bernoulli(p[state]),
                obs=torch.tensor(1.0 if data else 0.0),
            )

        @pyro.markov
        def model1(data, state=0, address=""):
            if isinstance(data, bool):
                model_leaf(data, state, address)
            else:
                p = pyro.param("p_branch", torch.ones(10, 10))
                for branch, letter in zip(data, "abcdefg"):
                    next_state = pyro.sample(
                        "branch_{}".format(address + letter),
                        dist.Categorical(p[state]),
                        infer={"enumerate": "parallel"},
                    )
                    model2(branch, next_state, address + letter)

        @pyro.markov
        def model2(data, state=0, address=""):
            if isinstance(data, bool):
                model_leaf(data, state, address)
            else:
                p = pyro.param("p_branch", torch.ones(10, 10))
                for branch, letter in zip(data, "abcdefg"):
                    next_state = pyro.sample(
                        "branch_{}".format(address + letter),
                        dist.Categorical(p[state]),
                        infer={"enumerate": "parallel"},
                    )
                    model1(branch, next_state, address + letter)

        return model1(**kwargs)

    assert_ok(model_, max_plate_nesting=0, data=data)


@pytest.mark.parametrize("max_plate_nesting", [0, 1, 2])
def test_enum_recycling_interleave(max_plate_nesting):
    def model():
        with pyro.markov() as m:
            with pyro.markov():
                with m:  # error here
                    pyro.sample(
                        "x",
                        dist.Categorical(torch.ones(4)),
                        infer={"enumerate": "parallel"},
                    )

    assert_ok(model, max_plate_nesting=max_plate_nesting)


@pytest.mark.parametrize("max_plate_nesting", [0, 1, 2])
@pytest.mark.parametrize("history", [2, 3])
def test_markov_history(max_plate_nesting, history):
    @infer.config_enumerate
    def model():
        p = pyro.param("p", 0.25 * torch.ones(2, 2))
        q = pyro.param("q", 0.25 * torch.ones(2))
        x_prev = torch.tensor(0)
        x_curr = torch.tensor(0)
        for t in pyro.markov(range(10), history=history):
            probs = p[x_prev, x_curr]
            x_prev, x_curr = (
                x_curr,
                pyro.sample("x_{}".format(t), dist.Bernoulli(probs)).long(),
            )
            pyro.sample(
                "y_{}".format(t), dist.Bernoulli(q[x_curr]), obs=torch.tensor(0.0)
            )

    assert_ok(model, max_plate_nesting=max_plate_nesting)
