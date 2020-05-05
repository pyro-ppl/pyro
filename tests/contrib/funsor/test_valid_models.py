# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
import contextlib
import logging
import os
from queue import LifoQueue

import pytest
import torch

import funsor
from funsor.domains import bint, reals
from funsor.tensor import Tensor

import pyro
from pyro.infer.enum import iter_discrete_escape, iter_discrete_extend
from pyro.ops.indexing import Vindex
from pyro.poutine import Trace
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_traceenum_requirements

import pyro.contrib.funsor
from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers.named_messenger import GlobalNamedMessenger
from pyro.contrib.funsor.handlers.runtime import _DIM_STACK

from pyroapi import distributions as dist
from pyroapi import infer, handlers, pyro, pyro_backend


funsor.set_backend("torch")

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
                guide_tr_pyro = handlers.trace(handlers.queue(
                    guide, q_pyro, escape_fn=iter_discrete_escape, extend_fn=iter_discrete_extend
                )).get_trace(**kwargs)
                tr_pyro = handlers.trace(handlers.replay(model, trace=guide_tr_pyro)).get_trace(**kwargs)

        with pyro_backend("contrib.funsor"):
            with handlers.enum(first_available_dim=-max_plate_nesting - 1):
                guide_tr_funsor = handlers.trace(handlers.queue(
                    guide, q_funsor, escape_fn=iter_discrete_escape, extend_fn=iter_discrete_extend
                )).get_trace(**kwargs)
                tr_funsor = handlers.trace(handlers.replay(model, trace=guide_tr_funsor)).get_trace(**kwargs)

        # make sure all dimensions were cleaned up
        assert len(_DIM_STACK._stack) == 1
        assert not _DIM_STACK.global_frame.name_to_dim and not _DIM_STACK.global_frame.dim_to_name
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
        node['infer']['_enumerate_symbol']: name
        for name, node in tr_pyro.nodes.items()
        if node['type'] == 'sample' and not node['is_observed']
        and node['infer'].get('enumerate') == 'parallel'
    }
    symbol_to_name.update({
        symbol: name for name, symbol in tr_pyro.plate_to_symbol.items()})

    if _NAMED_TEST_STRENGTH >= 1:
        # coarser check: enumeration requirements satisfied
        check_traceenum_requirements(tr_pyro, Trace())
        check_traceenum_requirements(tr_funsor, Trace())
        try:
            # coarser check: number of elements and squeezed shapes
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node['type'] != 'sample':
                    continue
                funsor_node = tr_funsor.nodes[name]
                assert pyro_node['packed']['log_prob'].numel() == funsor_node['log_prob'].numel()
                assert pyro_node['packed']['log_prob'].shape == funsor_node['log_prob'].squeeze().shape
                assert frozenset(f for f in pyro_node['cond_indep_stack'] if f.vectorized) == \
                    frozenset(f for f in funsor_node['cond_indep_stack'] if f.vectorized)
        except AssertionError:
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node['type'] != 'sample':
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_packed_shape = pyro_node['packed']['log_prob'].shape
                funsor_packed_shape = funsor_node['log_prob'].squeeze().shape
                if pyro_packed_shape != funsor_packed_shape:
                    err_str = f"==> (dep mismatch) {name}"
                else:
                    err_str = name
                print(err_str, f"Pyro: {pyro_packed_shape} vs Funsor: {funsor_packed_shape}")
            raise

    if _NAMED_TEST_STRENGTH >= 2:
        try:
            # medium check: unordered packed shapes match
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node['type'] != 'sample':
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_names = frozenset(symbol_to_name[d] for d in pyro_node['packed']['log_prob']._pyro_dims)
                funsor_names = frozenset(funsor_node['funsor']['log_prob'].inputs)
                try:
                    assert pyro_names == funsor_names
                except AssertionError:
                    assert pyro_names == frozenset(name.replace('__PARTICLES', '') for name in funsor_names)
        except AssertionError:
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node['type'] != 'sample':
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_names = frozenset(symbol_to_name[d] for d in pyro_node['packed']['log_prob']._pyro_dims)
                funsor_names = frozenset(funsor_node['funsor']['log_prob'].inputs)
                if pyro_names != funsor_names:
                    err_str = f"==> (packed mismatch) {name}"
                else:
                    err_str = name
                print(err_str, f"Pyro: {sorted(tuple(pyro_names))} vs Funsor: {sorted(tuple(funsor_names))}")
            raise

    if _NAMED_TEST_STRENGTH >= 3:
        try:
            # finer check: exact match with unpacked Pyro shapes
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node['type'] != 'sample':
                    continue
                funsor_node = tr_funsor.nodes[name]
                assert pyro_node['log_prob'].shape == funsor_node['log_prob'].shape
                assert pyro_node['value'].shape == funsor_node['value'].shape
        except AssertionError:
            for name, pyro_node in tr_pyro.nodes.items():
                if pyro_node['type'] != 'sample':
                    continue
                funsor_node = tr_funsor.nodes[name]
                pyro_shape = pyro_node['log_prob'].shape
                funsor_shape = funsor_node['log_prob'].shape
                if pyro_shape != funsor_shape:
                    err_str = f"==> (unpacked mismatch) {name}"
                else:
                    err_str = name
                print(err_str, f"Pyro: {pyro_shape} vs Funsor: {funsor_shape}")
            raise


def test_iteration():

    def testing():
        for i in pyro.markov(range(5)):
            v1 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{i}", bint(2))]), 'real'))
            v2 = to_data(Tensor(torch.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
            fv1 = to_funsor(v1, reals())
            fv2 = to_funsor(v2, reals())
            print(i, v1.shape)  # shapes should alternate
            if i % 2 == 0:
                assert v1.shape == (2,)
            else:
                assert v1.shape == (2, 1, 1)
            assert v2.shape == (2, 1)
            print(i, fv1.inputs)
            print('a', v2.shape)  # shapes should stay the same
            print('a', fv2.inputs)

    with pyro_backend("contrib.funsor"), GlobalNamedMessenger():
        testing()


def test_nesting():

    def testing():

        with pyro.markov():
            v1 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{1}", bint(2))]), 'real'))
            print(1, v1.shape)  # shapes should alternate
            assert v1.shape == (2,)

            with pyro.markov():
                v2 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{2}", bint(2))]), 'real'))
                print(2, v2.shape)  # shapes should alternate
                assert v2.shape == (2, 1)

                with pyro.markov():
                    v3 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{3}", bint(2))]), 'real'))
                    print(3, v3.shape)  # shapes should alternate
                    assert v3.shape == (2,)

                    with pyro.markov():
                        v4 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{4}", bint(2))]), 'real'))
                        print(4, v4.shape)  # shapes should alternate

                        assert v4.shape == (2, 1)

    with pyro_backend("contrib.funsor"), GlobalNamedMessenger():
        testing()


def test_staggered():

    def testing():
        for i in pyro.markov(range(12)):
            if i % 4 == 0:
                v2 = to_data(Tensor(torch.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
                fv2 = to_funsor(v2, reals())
                assert v2.shape == (2,)
                print('a', v2.shape)
                print('a', fv2.inputs)

    with pyro_backend("contrib.funsor"), GlobalNamedMessenger():
        testing()


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


@pytest.mark.xfail(reason="Pyro not handling mixed parallel/sequential enum well?")
@pytest.mark.parametrize("history", [1, 2, 3])
def test_enum_recycling_chain_iter_interleave_parallel_sequential(history):

    def model():
        p = torch.tensor([[0.2, 0.8], [0.1, 0.9]])

        xs = [0]
        for t in pyro.markov(range(10), history=history):
            xs.append(pyro.sample("x_{}".format(t), dist.Categorical(p[xs[-1]]),
                                  infer={"enumerate": ("sequential", "parallel")[t % 2]}))
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


@pytest.mark.parametrize('use_vindex', [False, True])
@pytest.mark.parametrize('markov', [False, True])
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
            pyro.sample("z_{}".format(t), dist.Categorical(probs),
                        obs=torch.tensor(0.))

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
@pytest.mark.parametrize('use_vindex', [False, True])
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
                    probs = p[x[i - 1, j].unsqueeze(-1),
                              x[i, j - 1].unsqueeze(-1), ind]
                x[i, j] = pyro.sample("x_{}_{}".format(i, j),
                                      dist.Categorical(probs))

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2])
@pytest.mark.parametrize("depth", [3, 5, 7])
@pytest.mark.parametrize('history', [1, 2])
def test_enum_recycling_reentrant_history(max_plate_nesting, depth, history):
    data = (True, False)
    for i in range(depth):
        data = (data, data, False)

    def model_(**kwargs):

        @pyro.markov(history=history)
        def model(data, state=0, address=""):
            if isinstance(data, bool):
                p = pyro.param("p_leaf", torch.ones(10))
                pyro.sample("leaf_{}".format(address),
                            dist.Bernoulli(p[state]),
                            obs=torch.tensor(1. if data else 0.))
            else:
                assert isinstance(data, tuple)
                p = pyro.param("p_branch", torch.ones(10, 10))
                for branch, letter in zip(data, "abcdefg"):
                    next_state = pyro.sample("branch_{}".format(address + letter),
                                             dist.Categorical(p[state]),
                                             infer={"enumerate": "parallel"})
                    model(branch, next_state, address + letter)

        return model(**kwargs)

    assert_ok(model_, max_plate_nesting=max_plate_nesting, data=data)


@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2])
@pytest.mark.parametrize("depth", [3, 5, 7])
def test_enum_recycling_mutual_recursion(max_plate_nesting, depth):
    data = (True, False)
    for i in range(depth):
        data = (data, data, False)

    def model_(**kwargs):

        def model_leaf(data, state=0, address=""):
            p = pyro.param("p_leaf", torch.ones(10))
            pyro.sample("leaf_{}".format(address),
                        dist.Bernoulli(p[state]),
                        obs=torch.tensor(1. if data else 0.))

        @pyro.markov
        def model1(data, state=0, address=""):
            if isinstance(data, bool):
                model_leaf(data, state, address)
            else:
                p = pyro.param("p_branch", torch.ones(10, 10))
                for branch, letter in zip(data, "abcdefg"):
                    next_state = pyro.sample("branch_{}".format(address + letter),
                                             dist.Categorical(p[state]),
                                             infer={"enumerate": "parallel"})
                    model2(branch, next_state, address + letter)

        @pyro.markov
        def model2(data, state=0, address=""):
            if isinstance(data, bool):
                model_leaf(data, state, address)
            else:
                p = pyro.param("p_branch", torch.ones(10, 10))
                for branch, letter in zip(data, "abcdefg"):
                    next_state = pyro.sample("branch_{}".format(address + letter),
                                             dist.Categorical(p[state]),
                                             infer={"enumerate": "parallel"})
                    model1(branch, next_state, address + letter)

        return model1(**kwargs)

    assert_ok(model_, max_plate_nesting=0, data=data)


@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2])
def test_enum_recycling_interleave(max_plate_nesting):

    def model():
        with pyro.markov() as m:
            with pyro.markov():
                with m:  # error here
                    pyro.sample("x", dist.Categorical(torch.ones(4)),
                                infer={"enumerate": "parallel"})

    assert_ok(model, max_plate_nesting=max_plate_nesting)


@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2])
@pytest.mark.parametrize('history', [2, 3])
def test_markov_history(max_plate_nesting, history):

    @infer.config_enumerate
    def model():
        p = pyro.param("p", 0.25 * torch.ones(2, 2))
        q = pyro.param("q", 0.25 * torch.ones(2))
        x_prev = torch.tensor(0)
        x_curr = torch.tensor(0)
        for t in pyro.markov(range(10), history=history):
            probs = p[x_prev, x_curr]
            x_prev, x_curr = x_curr, pyro.sample("x_{}".format(t), dist.Bernoulli(probs)).long()
            pyro.sample("y_{}".format(t), dist.Bernoulli(q[x_curr]),
                        obs=torch.tensor(0.))

    assert_ok(model, max_plate_nesting=max_plate_nesting)


@pytest.mark.parametrize('enumerate_', [None, "parallel", "sequential"])
def test_enum_discrete_non_enumerated_plate_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})

        with pyro.plate("non_enum", 2):
            a = pyro.sample("a", dist.Bernoulli(0.5), infer={'enumerate': None})

        p = (1.0 + a.sum(-1)) / (2.0 + a.shape[0])  # introduce dependency of b on a

        with pyro.plate("enum_1", 3):
            pyro.sample("b", dist.Bernoulli(p), infer={'enumerate': enumerate_})

    assert_ok(model, max_plate_nesting=1)


@pytest.mark.parametrize("plate_dims", [
    (None, None, None, None),
    (-3, None, None, None),
    (None, -3, None, None),
    (-2, -3, None, None),
])
def test_plate_dim_allocation_ok(plate_dims):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_outer", 5, dim=plate_dims[0]):
            pyro.sample("x", dist.Bernoulli(p))
            with pyro.plate("plate_inner_1", 6, dim=plate_dims[1]):
                pyro.sample("y", dist.Bernoulli(p))
                with pyro.plate("plate_inner_2", 7, dim=plate_dims[2]):
                    pyro.sample("z", dist.Bernoulli(p))
                    with pyro.plate("plate_inner_3", 8, dim=plate_dims[3]):
                        pyro.sample("q", dist.Bernoulli(p))

    assert_ok(model, max_plate_nesting=4)


@pytest.mark.parametrize("tmc_strategy", [None, "diagonal"])
@pytest.mark.parametrize("subsampling", [False, True])
@pytest.mark.parametrize("reuse_plate", [False, True])
def test_enum_recycling_plate(subsampling, reuse_plate, tmc_strategy):

    @infer.config_enumerate(default="parallel", tmc=tmc_strategy, num_samples=2 if tmc_strategy else None)
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        q = pyro.param("q", torch.tensor([0.5, 0.5]))
        plate_x = pyro.plate("plate_x", 4, subsample_size=3 if subsampling else None, dim=-1)
        plate_y = pyro.plate("plate_y", 5, subsample_size=3 if subsampling else None, dim=-1)
        plate_z = pyro.plate("plate_z", 6, subsample_size=3 if subsampling else None, dim=-2)

        a = pyro.sample("a", dist.Bernoulli(q[0])).long()
        w = 0
        for i in pyro.markov(range(5)):
            w = pyro.sample("w_{}".format(i), dist.Categorical(p[w]))

        with plate_x:
            b = pyro.sample("b", dist.Bernoulli(q[a])).long()
            x = 0
            for i in pyro.markov(range(6)):
                x = pyro.sample("x_{}".format(i), dist.Categorical(p[x]))

        with plate_y:
            c = pyro.sample("c", dist.Bernoulli(q[a])).long()
            y = 0
            for i in pyro.markov(range(7)):
                y = pyro.sample("y_{}".format(i), dist.Categorical(p[y]))

        with plate_z:
            d = pyro.sample("d", dist.Bernoulli(q[a])).long()
            z = 0
            for i in pyro.markov(range(8)):
                z = pyro.sample("z_{}".format(i), dist.Categorical(p[z]))

        with plate_x, plate_z:
            # this part is tricky: how do we know to preserve b's dimension?
            # also, how do we know how to make b and d have different dimensions?
            e = pyro.sample("e", dist.Bernoulli(q[b if reuse_plate else a])).long()
            xz = 0
            for i in pyro.markov(range(9)):
                xz = pyro.sample("xz_{}".format(i), dist.Categorical(p[xz]))

        return a, b, c, d, e

    assert_ok(model, max_plate_nesting=2)


@pytest.mark.parametrize("enumerate_", [None, "parallel", "sequential"])
@pytest.mark.parametrize("reuse_plate", [True, False])
def test_enum_discrete_plates_dependency_ok(enumerate_, reuse_plate):

    @infer.config_enumerate(default=enumerate_)
    def model():
        x_plate = pyro.plate("x_plate", 10, dim=-1)
        y_plate = pyro.plate("y_plate", 11, dim=-2)
        q = pyro.param("q", torch.tensor([0.5, 0.5]))
        pyro.sample("a", dist.Bernoulli(0.5))
        with x_plate:
            b = pyro.sample("b", dist.Bernoulli(0.5)).long()
        with y_plate:
            # Note that it is difficult to check that c does not depend on b.
            c = pyro.sample("c", dist.Bernoulli(0.5)).long()
        with x_plate, y_plate:
            pyro.sample("d", dist.Bernoulli(Vindex(q)[b] if reuse_plate else 0.5))

        assert c.shape != b.shape or enumerate_ == "sequential"

    assert_ok(model, max_plate_nesting=2)


@pytest.mark.parametrize('subsampling', [False, True])
@pytest.mark.parametrize('enumerate_', [None, "parallel", "sequential"])
def test_enum_discrete_plate_shape_broadcasting_ok(subsampling, enumerate_):

    @infer.config_enumerate(default=enumerate_)
    def model():
        x_plate = pyro.plate("x_plate", 5, subsample_size=2 if subsampling else None, dim=-1)
        y_plate = pyro.plate("y_plate", 6, subsample_size=3 if subsampling else None, dim=-2)
        with pyro.plate("num_particles", 50, dim=-3):
            with x_plate:
                b = pyro.sample("b", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            with y_plate:
                c = pyro.sample("c", dist.Bernoulli(0.5))
            with x_plate, y_plate:
                d = pyro.sample("d", dist.Bernoulli(b))

        # check shapes
        if enumerate_ == "parallel":
            assert b.shape == (50, 1, x_plate.subsample_size)
            assert c.shape == (2, 1, 1, 1)
            assert d.shape == (2, 1, 1, 1, 1)
        elif enumerate_ == "sequential":
            assert b.shape == (50, 1, x_plate.subsample_size)
            assert c.shape in ((), (1, 1, 1))  # both are valid
            assert d.shape in ((), (1, 1, 1))  # both are valid
        else:
            assert b.shape == (50, 1, x_plate.subsample_size)
            assert c.shape == (50, y_plate.subsample_size, 1)
            assert d.shape == (50, y_plate.subsample_size, x_plate.subsample_size)

    assert_ok(model, guide=model, max_plate_nesting=3)


@pytest.mark.parametrize('subsampling', [False, True])
@pytest.mark.parametrize('enumerate_', [None, "parallel", "sequential"])
def test_enum_discrete_iplate_plate_dependency_ok(subsampling, enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate("plate", 10, subsample_size=4 if subsampling else None)
        for i in pyro.plate("iplate", 10, subsample=torch.arange(3) if subsampling else None):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))
            with inner_plate:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5),
                            infer={'enumerate': enumerate_})

    assert_ok(model, max_plate_nesting=1)


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("num_samples", [None, 2])
def test_plate_subsample_primitive_ok(subsample_size, num_samples):

    @infer.config_enumerate(num_samples=num_samples, tmc="diagonal")
    def model():
        with pyro.plate("plate", 10, subsample_size=subsample_size, dim=None):
            p0 = torch.tensor(0.)
            p0 = pyro.subsample(p0, event_dim=0)
            assert p0.shape == ()
            p = 0.5 * torch.ones(10)
            p = pyro.subsample(p, event_dim=0)
            assert len(p) == (subsample_size if subsample_size else 10)
            pyro.sample("x", dist.Bernoulli(p))

    assert_ok(model, max_plate_nesting=1)


@pytest.mark.xfail(reason="empty samples incompatible with current Pyro")
def test_enum_iplate_iplate_ok_1():

    @infer.config_enumerate
    def model(data=None):
        probs_a = torch.tensor([0.45, 0.55])
        probs_b = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        probs_c = torch.tensor([[0.75, 0.25], [0.55, 0.45]])
        probs_d = torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

        @handlers.trace
        def model_():
            b_axis = pyro.plate("b_axis", 2)
            c_axis = pyro.plate("c_axis", 2)
            a = pyro.sample("a", dist.Categorical(probs_a))
            b = [pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a])) for i in b_axis]
            c = [pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis]
            for i in b_axis:
                for j in c_axis:
                    b_i, c_j = pyro.sample("b_{}".format(i)), pyro.sample("c_{}".format(j))
                    pyro.sample("d_{}_{}".format(i, j),
                                dist.Categorical(Vindex(probs_d)[b_i, c_j]),
                                obs=data[i, j])

        return model_()

    data = torch.tensor([[0, 1], [0, 0]])
    assert_ok(model, max_plate_nesting=1, data=data)


def test_enum_iplate_iplate_ok_2():

    @infer.config_enumerate
    def model(data=None):
        probs_a = torch.tensor([0.45, 0.55])
        probs_b = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        probs_c = torch.tensor([[0.75, 0.25], [0.55, 0.45]])
        probs_d = torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a])) for i in b_axis]
        c = [pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis]
        for i in b_axis:
            for j in c_axis:
                pyro.sample("d_{}_{}".format(i, j),
                            dist.Categorical(Vindex(probs_d)[b[i], c[j]]),
                            obs=data[i, j])

    data = torch.tensor([[0, 1], [0, 0]])
    assert_ok(model, max_plate_nesting=1, data=data)


def test_enum_plate_iplate_ok():

    @infer.config_enumerate
    def model(data=None):
        probs_a = torch.tensor([0.45, 0.55])
        probs_b = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        probs_c = torch.tensor([[0.75, 0.25], [0.55, 0.45]])
        probs_d = torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        c = [pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis]
        with b_axis:
            for j in c_axis:
                pyro.sample("d_{}".format(j),
                            dist.Categorical(Vindex(probs_d)[b, c[j]]),
                            obs=data[:, j])

    data = torch.tensor([[0, 1], [0, 0]])
    assert_ok(model, max_plate_nesting=1, data=data)


def test_enum_iplate_plate_ok():

    @infer.config_enumerate
    def model(data=None):
        probs_a = torch.tensor([0.45, 0.55])
        probs_b = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        probs_c = torch.tensor([[0.75, 0.25], [0.55, 0.45]])
        probs_d = torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a])) for i in b_axis]
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        for i in b_axis:
            with c_axis:
                pyro.sample("d_{}".format(i),
                            dist.Categorical(Vindex(probs_d)[b[i], c]),
                            obs=data[i])

    data = torch.tensor([[0, 1], [0, 0]])
    assert_ok(model, max_plate_nesting=1, data=data)
