# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, defaultdict
import contextlib
import logging

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate
from pyro.ops.indexing import Vindex

from pyro.contrib.funsor import to_data, to_funsor, markov
from pyro.contrib.funsor.named_messenger import GlobalNamedMessenger
from pyro.contrib.funsor.enum_messenger import EnumMessenger

logger = logging.getLogger(__name__)

_ENUM_BACKEND_VERSION = "pyro"


@contextlib.contextmanager
def toggle_backend(backend):
    global _ENUM_BACKEND_VERSION
    old_version = _ENUM_BACKEND_VERSION
    _ENUM_BACKEND_VERSION = backend
    try:
        yield
    finally:
        _ENUM_BACKEND_VERSION = old_version


def pyro_plate(*args, **kwargs):
    return pyro.plate(*args, **kwargs)  # PlateMessenger(*args, **kwargs)


def pyro_markov(*args, **kwargs):
    global _ENUM_BACKEND_VERSION
    return (markov if _ENUM_BACKEND_VERSION == "funsor" else pyro.markov)(*args, **kwargs)


def assert_ok(model, max_plate_nesting=None, **kwargs):
    """
    Assert that enumeration runs...
    """
    pyro.clear_param_store()
    with toggle_backend("pyro"), poutine.trace() as tr_pyro:
        with poutine.enum(first_available_dim=-max_plate_nesting - 1):
            model(**kwargs)

    with toggle_backend("funsor"), poutine.trace() as tr_funsor:
        with EnumMessenger(first_available_dim=-max_plate_nesting - 1):
            with markov():
                model(**kwargs)

    assert tr_pyro.trace.nodes.keys() == tr_funsor.trace.nodes.keys()
    tr_pyro.trace.compute_log_prob()
    tr_funsor.trace.compute_log_prob()
    tr_pyro.trace.pack_tensors()

    symbol_to_name = {
        node['infer']['_enumerate_symbol']: name
        for name, node in tr_pyro.trace.nodes.items()
        if node['type'] == 'sample' and not node['is_observed']
        and node['infer'].get('enumerate') == 'parallel'
    }

    try:
        # coarser check: number of elements and squeezed shapes
        for name, pyro_node in tr_pyro.trace.nodes.items():
            if pyro_node['type'] != 'sample':
                continue
            funsor_node = tr_funsor.trace.nodes[name]
            assert pyro_node['packed']['log_prob'].numel() == funsor_node['log_prob'].numel()
            assert pyro_node['packed']['log_prob'].shape == funsor_node['log_prob'].squeeze().shape

        # medium check: unordered packed shapes match
        for name, pyro_node in tr_pyro.trace.nodes.items():
            if pyro_node['type'] != 'sample':
                continue
            funsor_node = tr_funsor.trace.nodes[name]
            pyro_names = frozenset(symbol_to_name[d] for d in pyro_node['packed']['log_prob']._pyro_dims)
            funsor_names = frozenset(funsor_node['infer']['funsor_log_prob'].inputs)
            assert pyro_names == funsor_names

        # finer check: exact match with unpacked Pyro shapes
        for name, pyro_node in tr_pyro.trace.nodes.items():
            if pyro_node['type'] != 'sample':
                continue
            assert pyro_node['log_prob'].shape == funsor_node['log_prob'].shape
            assert pyro_node['value'].shape == funsor_node['value'].shape
    except AssertionError:
        for name, pyro_node in tr_pyro.trace.nodes.items():
            if pyro_node['type'] != 'sample':
                continue
            funsor_node = tr_funsor.trace.nodes[name]
            print(name, pyro_node['log_prob'].shape, funsor_node['log_prob'].shape)
        raise


def test_iteration():
    import funsor; funsor.set_backend("torch")  # noqa: E702
    from funsor.domains import bint, reals
    from funsor.tensor import Tensor

    def testing():
        for i in pyro_markov(range(5)):
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

    with toggle_backend("funsor"), GlobalNamedMessenger():
        testing()


def test_nesting():
    import funsor; funsor.set_backend("torch")  # noqa: E702
    from funsor.domains import bint
    from funsor.tensor import Tensor

    def testing():

        with pyro_markov():
            v1 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{1}", bint(2))]), 'real'))
            print(1, v1.shape)  # shapes should alternate
            assert v1.shape == (2,)

            with pyro_markov():
                v2 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{2}", bint(2))]), 'real'))
                print(2, v2.shape)  # shapes should alternate
                assert v2.shape == (2, 1)

                with pyro_markov():
                    v3 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{3}", bint(2))]), 'real'))
                    print(3, v3.shape)  # shapes should alternate
                    assert v3.shape == (2,)

                    with pyro_markov():
                        v4 = to_data(Tensor(torch.ones(2), OrderedDict([(f"{4}", bint(2))]), 'real'))
                        print(4, v4.shape)  # shapes should alternate

                        assert v4.shape == (2, 1)

    with toggle_backend("funsor"), GlobalNamedMessenger():
        testing()


def test_staggered():
    import funsor; funsor.set_backend("torch")  # noqa: E702
    from funsor.domains import bint, reals
    from funsor.tensor import Tensor

    def testing():
        for i in pyro_markov(range(12)):
            if i % 4 == 0:
                v2 = to_data(Tensor(torch.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
                fv2 = to_funsor(v2, reals())
                assert v2.shape == (2,)
                print('a', v2.shape)
                print('a', fv2.inputs)

    with toggle_backend("funsor"), GlobalNamedMessenger():
        testing()


def test_enum_recycling_chain():

    @config_enumerate
    def model():
        p = torch.tensor([[0.2, 0.8], [0.1, 0.9]])

        x = 0
        for t in pyro_markov(range(100)):
            x = pyro.sample("x_{}".format(t), dist.Categorical(p[x]))
            assert x.dim() <= 2

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize('use_vindex', [False, True])
@pytest.mark.parametrize('markov', [False, True])
def test_enum_recycling_dbn(markov, use_vindex):
    #    x --> x --> x  enum "state"
    # y  |  y  |  y  |  enum "occlusion"
    #  \ |   \ |   \ |
    #    z     z     z  obs

    @config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        q = pyro.param("q", torch.ones(2))
        r = pyro.param("r", torch.ones(3, 2, 4))

        x = 0
        times = pyro_markov(range(100)) if markov else range(11)
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

    @config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        x = pyro.sample("x", dist.Categorical(p[0]))
        y = x
        for i in pyro_markov(range(10)):
            y = pyro.sample("y_{}".format(i), dist.Categorical(p[y]))
            z = y
            for j in pyro_markov(range(10)):
                z = pyro.sample("z_{}_{}".format(i, j), dist.Categorical(p[z]))

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize('use_vindex', [False, True])
def test_enum_recycling_grid(use_vindex):
    #  x---x---x---x    -----> i
    #  |   |   |   |   |
    #  x---x---x---x   |
    #  |   |   |   |   V
    #  x---x---x--(x)  j
    #  |   |   |   |
    #  x---x--(x)--x <-- what can this depend on?

    @config_enumerate
    def model():
        p = pyro.param("p_leaf", torch.ones(2, 2, 2))
        x = defaultdict(lambda: torch.tensor(0))
        y_axis = pyro_markov(range(4), keep=True)
        for i in pyro_markov(range(4)):
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


def test_enum_recycling_reentrant():
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    def model_(**kwargs):

        @pyro_markov
        def model(data, state=0, address=""):
            if isinstance(data, bool):
                p = pyro.param("p_leaf", torch.ones(10))
                pyro.sample("leaf_{}".format(address),
                            dist.Bernoulli(p[state]),
                            obs=torch.tensor(1. if data else 0.))
            else:
                p = pyro.param("p_branch", torch.ones(10, 10))
                for branch, letter in zip(data, "abcdefg"):
                    next_state = pyro.sample("branch_{}".format(address + letter),
                                             dist.Categorical(p[state]),
                                             infer={"enumerate": "parallel"})
                    model(branch, next_state, address + letter)

        return model(**kwargs)

    assert_ok(model_, max_plate_nesting=0, data=data)


@pytest.mark.parametrize('history', [1, 2])
def test_enum_recycling_reentrant_history(history):
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    def model_(**kwargs):

        @pyro_markov(history=history)
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

    assert_ok(model_, max_plate_nesting=0, data=data)


def test_enum_recycling_mutual_recursion():
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    def model_(**kwargs):

        def model_leaf(data, state=0, address=""):
            p = pyro.param("p_leaf", torch.ones(10))
            pyro.sample("leaf_{}".format(address),
                        dist.Bernoulli(p[state]),
                        obs=torch.tensor(1. if data else 0.))

        @pyro_markov
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

        @pyro_markov
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


def test_enum_recycling_interleave():

    def model():
        with pyro_markov() as m:
            with pyro_markov():
                with m:  # error here
                    pyro.sample("x", dist.Categorical(torch.ones(4)),
                                infer={"enumerate": "parallel"})

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.parametrize('history', [2, 3])
def test_markov_history(history):

    @config_enumerate
    def model():
        p = pyro.param("p", 0.25 * torch.ones(2, 2))
        q = pyro.param("q", 0.25 * torch.ones(2))
        x_prev = torch.tensor(0)
        x_curr = torch.tensor(0)
        for t in pyro_markov(range(10), history=history):
            probs = p[x_prev, x_curr]
            x_prev, x_curr = x_curr, pyro.sample("x_{}".format(t), dist.Bernoulli(probs)).long()
            pyro.sample("y_{}".format(t), dist.Bernoulli(q[x_curr]),
                        obs=torch.tensor(0.))

    assert_ok(model, max_plate_nesting=0)


@pytest.mark.xfail(reason="plate not supported yet")
def test_enum_recycling_plate():

    @config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        q = pyro.param("q", torch.tensor([0.5, 0.5]))
        plate_x = pyro.plate("plate_x", 2, dim=-1)
        plate_y = pyro.plate("plate_y", 3, dim=-1)
        plate_z = pyro.plate("plate_z", 4, dim=-2)

        a = pyro.sample("a", dist.Bernoulli(q[0])).long()
        w = 0
        for i in pyro_markov(range(5)):
            w = pyro.sample("w_{}".format(i), dist.Categorical(p[w]))

        with plate_x:
            b = pyro.sample("b", dist.Bernoulli(q[a])).long()
            x = 0
            for i in pyro_markov(range(6)):
                x = pyro.sample("x_{}".format(i), dist.Categorical(p[x]))

        with plate_y:
            c = pyro.sample("c", dist.Bernoulli(q[a])).long()
            y = 0
            for i in pyro_markov(range(7)):
                y = pyro.sample("y_{}".format(i), dist.Categorical(p[y]))

        with plate_z:
            d = pyro.sample("d", dist.Bernoulli(q[a])).long()
            z = 0
            for i in pyro_markov(range(8)):
                z = pyro.sample("z_{}".format(i), dist.Categorical(p[z]))

        with plate_x, plate_z:
            e = pyro.sample("e", dist.Bernoulli(q[b])).long()
            xz = 0
            for i in pyro_markov(range(9)):
                xz = pyro.sample("xz_{}".format(i), dist.Categorical(p[xz]))

        return a, b, c, d, e

    assert_ok(model, max_plate_nesting=2)
