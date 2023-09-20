# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.infer.inspect import _deep_merge, get_dependencies, get_model_relations


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_get_dependencies(grad_enabled):
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", NonreparameterizedNormal(a, 0))
        c = pyro.sample("c", dist.Normal(b, 1))
        d = pyro.sample("d", dist.Normal(a, c.exp()))

        e = pyro.sample("e", dist.Normal(0, 1))
        f = pyro.sample("f", dist.Normal(0, 1))
        g = pyro.sample("g", dist.Bernoulli(logits=e + f), obs=torch.tensor(0.0))

        with pyro.plate("p", len(data)):
            d_ = d.detach()  # this results in a known failure
            h = pyro.sample("h", dist.Normal(c, d_.exp()))
            i = pyro.deterministic("i", h + 1)
            j = pyro.sample("j", dist.Delta(h + 1), obs=h + 1)
            k = pyro.sample("k", dist.Normal(a, j.exp()), obs=data)

        return [a, b, c, d, e, f, g, h, i, j, k]

    data = torch.randn(3)
    with torch.set_grad_enabled(grad_enabled):
        actual = get_dependencies(model, (data,))
    _ = set()
    expected = {
        "prior_dependencies": {
            "a": {"a": _},
            "b": {"b": _, "a": _},
            "c": {"c": _, "b": _},
            "d": {"d": _, "c": _, "a": _},
            "e": {"e": _},
            "f": {"f": _},
            "g": {"g": _, "e": _, "f": _},
            "h": {"h": _, "c": _, "d": _},
            "k": {"k": _, "a": _, "h": _},
        },
        "posterior_dependencies": {
            "a": {"a": _, "b": _, "c": _, "d": _, "h": _, "k": _},
            "b": {"b": _, "c": _},
            "c": {"c": _, "d": _, "h": _},
            "d": {"d": _, "h": _},
            "e": {"e": _, "g": _, "f": _},
            "f": {"f": _, "g": _},
            "h": {"h": _, "k": _},
        },
    }
    assert actual == expected


def test_docstring_example_1():
    def model_1():
        a = pyro.sample("a", dist.Normal(0, 1))
        pyro.sample("b", dist.Normal(a, 1), obs=torch.tensor(0.0))

    actual = get_dependencies(model_1)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set()},
        },
    }
    assert actual == expected


def test_docstring_example_2():
    def model_2():
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.LogNormal(0, 1))
        c = pyro.sample("c", dist.Normal(a, b))
        pyro.sample("d", dist.Normal(c, 1), obs=torch.tensor(0.0))

    actual = get_dependencies(model_2)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"b": set()},
            "c": {"a": set(), "b": set(), "c": set()},
            "d": {"c": set(), "d": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set(), "c": set()},
            "b": {"b": set(), "c": set()},
            "c": {"c": set(), "d": set()},
        },
    }
    assert actual == expected


def test_docstring_example_3():
    def model_3():
        with pyro.plate("p", 5):
            a = pyro.sample("a", dist.Normal(0, 1))
        pyro.sample("b", dist.Normal(a.sum(), 1), obs=torch.tensor(0.0))

    actual = get_dependencies(model_3)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
        },
        "posterior_dependencies": {
            "a": {"a": {"p"}, "b": set()},
        },
    }
    assert actual == expected


def test_factor():
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        pyro.factor("b", torch.tensor(0.0))
        pyro.factor("c", a)

    actual = get_dependencies(model)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"b": set()},
            "c": {"c": set(), "a": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "c": set()},
        },
    }
    assert actual == expected


def test_discrete_obs():
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a[..., None], torch.ones(3)).to_event(1))
        c = pyro.sample(
            "c", dist.MultivariateNormal(torch.zeros(3) + a[..., None], torch.eye(3))
        )
        with pyro.plate("i", 2):
            d = pyro.sample("d", dist.Dirichlet((b + c).exp()))
            pyro.sample("e", dist.Categorical(logits=d), obs=torch.tensor([0, 0]))
        return a, b, c, d

    actual = get_dependencies(model)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
            "c": {"a": set(), "c": set()},
            "d": {"b": set(), "c": set(), "d": set()},
            "e": {"d": set(), "e": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set(), "c": set()},
            "b": {"b": set(), "c": set(), "d": set()},
            "c": {"c": set(), "d": set()},
            "d": {"d": set(), "e": set()},
        },
    }
    assert actual == expected


def test_discrete():
    def model():
        a = pyro.sample("a", dist.Dirichlet(torch.ones(3)))
        b = pyro.sample("b", dist.Categorical(a))
        c = pyro.sample("c", dist.Normal(torch.zeros(3), 1).to_event(1))
        d = pyro.sample("d", dist.Poisson(c[b].exp()))
        pyro.sample("e", dist.Normal(d, 1), obs=torch.ones(()))

    actual = get_dependencies(model)
    expected = {
        "prior_dependencies": {
            "a": {"a": set()},
            "b": {"a": set(), "b": set()},
            "c": {"c": set()},
            "d": {"b": set(), "c": set(), "d": set()},
            "e": {"d": set(), "e": set()},
        },
        "posterior_dependencies": {
            "a": {"a": set(), "b": set()},
            "b": {"b": set(), "c": set(), "d": set()},
            "c": {"c": set(), "d": set()},
            "d": {"d": set(), "e": set()},
        },
    }
    assert actual == expected


def test_plate_coupling():
    #   x  x
    #    ||
    #    y
    #
    # This results in posterior dependency structure:
    #
    #     x x y
    #   x ? ? ?
    #   x ? ? ?

    def model(data):
        with pyro.plate("p", len(data)):
            x = pyro.sample("x", dist.Normal(0, 1))
        pyro.sample("y", dist.Normal(x.sum(), 1), obs=data.sum())

    data = torch.randn(2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": set()},
            "y": {"y": set(), "x": set()},
        },
        "posterior_dependencies": {
            "x": {"x": {"p"}, "y": set()},
        },
    }
    assert actual == expected


def test_plate_coupling_2():
    #   x x
    #     \\   y y
    #      \\ //
    #        z
    #
    # This results in posterior dependency structure:
    #
    #     x x y y z
    #   x ? ? ? ? ?
    #   x ? ? ? ? ?
    #   y     ? ? ?
    #   y     ? ? ?

    def model(data):
        with pyro.plate("p", len(data)):
            x = pyro.sample("x", dist.Normal(0, 1))
            y = pyro.sample("y", dist.Normal(0, 1))
        pyro.sample("z", dist.Normal(x.sum(), y.sum().exp()), obs=data.sum())

    data = torch.randn(2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": set()},
            "y": {"y": set()},
            "z": {"z": set(), "x": set(), "y": set()},
        },
        "posterior_dependencies": {
            "x": {"x": {"p"}, "y": {"p"}, "z": set()},
            "y": {"y": {"p"}, "z": set()},
        },
    }
    assert actual == expected


def test_plate_coupling_3():
    #    x x x x
    #     // \\
    #   y y   z z
    #
    # This results in posterior dependency structure:
    #
    #     x x y y z
    #   x ? ? ? ? ?
    #   x ? ? ? ? ?
    #   y     ? ? ?
    #   y     ? ? ?

    def model(data):
        i_plate = pyro.plate("i", data.shape[0], dim=-2)
        j_plate = pyro.plate("j", data.shape[1], dim=-1)
        with i_plate, j_plate:
            x = pyro.sample("x", dist.Normal(0, 1))
        with i_plate:
            pyro.sample("y", dist.Normal(x.sum(-1, True), 1), obs=data.sum(-1, True))
        with j_plate:
            pyro.sample("z", dist.Normal(x.sum(-2, True), 1), obs=data.sum(-2, True))

    data = torch.randn(3, 2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": set()},
            "y": {"y": set(), "x": set()},
            "z": {"z": set(), "x": set()},
        },
        "posterior_dependencies": {
            "x": {"x": {"i", "j"}, "y": set(), "z": set()},
        },
    }
    assert actual == expected


def test_plate_collider():
    #   x x    y y
    #     \\  //
    #      zzzz
    #
    # This results in posterior dependency structure:
    #
    #     x x y y z z z z
    #   x ?   ? ? ? ?
    #   x   ? ? ?     ? ?
    #   y     ?   ?   ?
    #   y       ?   ?   ?

    def model(data):
        i_plate = pyro.plate("i", data.shape[0], dim=-2)
        j_plate = pyro.plate("j", data.shape[1], dim=-1)

        with i_plate:
            x = pyro.sample("x", dist.Normal(0, 1))
        with j_plate:
            y = pyro.sample("y", dist.Normal(0, 1))
        with i_plate, j_plate:
            pyro.sample("z", dist.Normal(x, y.exp()), obs=data)

    data = torch.randn(3, 2)
    actual = get_dependencies(model, (data,))
    _ = set()
    expected = {
        "prior_dependencies": {
            "x": {"x": _},
            "y": {"y": _},
            "z": {"x": _, "y": _, "z": _},
        },
        "posterior_dependencies": {
            "x": {"x": _, "y": _, "z": _},
            "y": {"y": _, "z": _},
        },
    }
    assert actual == expected


def test_plate_dependency():
    #   w                              w
    #     \  x1 x2      unroll    x1  / \  x2
    #      \  || y1 y2  =====>  y1 | /   \ | y2
    #       \ || //               \|/     \|/
    #        z1 z2                z1       z2
    #
    # This allows posterior dependency structure:
    #
    #     w x x y y z z
    #   w ? ? ? ? ? ? ?
    #   x   ?   ?   ?
    #   x     ?   ?   ?
    #   y       ?   ?
    #   y         ?   ?

    def model(data):
        w = pyro.sample("w", dist.Normal(0, 1))
        with pyro.plate("p", len(data)):
            x = pyro.sample("x", dist.Normal(0, 1))
            y = pyro.sample("y", dist.Normal(0, 1))
            pyro.sample("z", dist.Normal(w + x + y, 1), obs=data)

    data = torch.rand(2)
    actual = get_dependencies(model, (data,))
    _ = set()
    expected = {
        "prior_dependencies": {
            "w": {"w": _},
            "x": {"x": _},
            "y": {"y": _},
            "z": {"w": _, "x": _, "y": _, "z": _},
        },
        "posterior_dependencies": {
            "w": {"w": _, "x": _, "y": _, "z": _},
            "x": {"x": _, "y": _, "z": _},
            "y": {"y": _, "z": _},
        },
    }
    assert actual == expected


def test_nested_plate_collider():
    # a a       b b
    #  a a     b b
    #    \\   //
    #      c c
    #       |
    #       d

    def model():
        plate_i = pyro.plate("i", 2, dim=-1)
        plate_j = pyro.plate("j", 3, dim=-2)
        plate_k = pyro.plate("k", 3, dim=-2)

        with plate_i:
            with plate_j:
                a = pyro.sample("a", dist.Normal(0, 1))
            with plate_k:
                b = pyro.sample("b", dist.Normal(0, 1))
            c = pyro.sample("c", dist.Normal(a.sum(0) + b.sum([0, 1]), 1))
        pyro.sample("d", dist.Normal(c.sum(), 1), obs=torch.zeros(()))

    actual = get_dependencies(model)
    _ = set()
    expected = {
        "prior_dependencies": {
            "a": {"a": _},
            "b": {"b": _},
            "c": {"c": _, "a": _, "b": _},
            "d": {"d": _, "c": _},
        },
        "posterior_dependencies": {
            "a": {"a": {"j"}, "b": _, "c": _},
            "b": {"b": {"k"}, "c": _},
            "c": {"c": {"i"}, "d": _},
        },
    }
    assert actual == expected


DEEP_MERGE_EXAMPLES = [
    ([True], True),
    ([False], False),
    ([True, True], True),
    ([True, False], None),
    ([False, False], False),
    ([{"a": True}], {"a": True}),
    ([{"a": True}, {"a": True}], {"a": True}),
    ([{"a": True}, {"a": False}], {"a": None}),
    (
        [
            {"a": True, "b": {"c": True}, "d": False, "e": 0},
            {"a": True, "b": {"c": True}, "d": True, "e": 1},
            {"a": True, "b": {"c": False}, "d": False, "e": 2},
            {"a": True, "b": {"c": True}, "d": False, "e": 3},
        ],
        {"a": True, "b": {"c": None}, "d": None, "e": 0},
    ),
]


@pytest.mark.parametrize("things, expected", DEEP_MERGE_EXAMPLES)
def test_deep_merge(things, expected):
    actual = _deep_merge(things)
    assert actual == expected


@pytest.mark.parametrize("include_deterministic", [True, False])
def test_get_model_relations(include_deterministic):
    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a, 1))
        c = pyro.sample("c", dist.Normal(a, b.exp()))
        d = pyro.sample("d", dist.Bernoulli(logits=c), obs=torch.tensor(0.0))

        with pyro.plate("p", len(data)):
            e = pyro.sample("e", dist.Normal(a, b.exp()))
            f = pyro.deterministic("f", e + 1)
            g = pyro.sample("g", dist.Delta(e + 1), obs=e + 1)
            h = pyro.sample("h", dist.Delta(e + 1))
            i = pyro.sample("i", dist.Normal(e, (f + g + h).exp()), obs=data)

        return [a, b, c, d, e, f, g, h, i]

    data = torch.randn(3)
    actual = get_model_relations(
        model,
        (data,),
        include_deterministic=include_deterministic,
    )

    if include_deterministic:
        expected = {
            "observed": ["d", "f", "g", "i"],
            "param_constraint": {},
            "plate_sample": {"p": ["e", "f", "g", "h", "i"]},
            "sample_dist": {
                "a": "Normal",
                "b": "Normal",
                "c": "Normal",
                "d": "Bernoulli",
                "e": "Normal",
                "f": "Deterministic",
                "g": "Delta",
                "h": "Delta",
                "i": "Normal",
            },
            "sample_param": {
                "a": [],
                "b": [],
                "c": [],
                "d": [],
                "e": [],
                "f": [],
                "g": [],
                "h": [],
                "i": [],
            },
            "sample_sample": {
                "a": [],
                "b": ["a"],
                "c": ["a", "b"],
                "d": ["c"],
                "e": ["a", "b"],
                "f": ["e"],
                "g": ["e"],
                "h": ["e"],
                "i": ["e", "f", "g", "h"],
            },
        }
    else:
        expected = {
            "sample_sample": {
                "a": [],
                "b": ["a"],
                "c": ["a", "b"],
                "d": ["c"],
                "e": ["a", "b"],
                "f": ["e"],
                "g": ["e"],
                "h": ["e"],
                "i": ["e"],
            },
            "sample_param": {
                "a": [],
                "b": [],
                "c": [],
                "d": [],
                "e": [],
                "f": [],
                "g": [],
                "h": [],
                "i": [],
            },
            "sample_dist": {
                "a": "Normal",
                "b": "Normal",
                "c": "Normal",
                "d": "Bernoulli",
                "e": "Normal",
                "f": "Deterministic",
                "g": "Delta",
                "h": "Delta",
                "i": "Normal",
            },
            "param_constraint": {},
            "plate_sample": {"p": ["e", "f", "g", "h", "i"]},
            "observed": ["d", "f", "g", "i"],
        }

    assert actual == expected
