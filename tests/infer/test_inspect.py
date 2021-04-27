# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.infer.inspect import get_dependencies


def test_get_dependencies():

    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", NonreparameterizedNormal(a, 0))
        c = pyro.sample("c", dist.Normal(b, 1))
        d = pyro.sample("d", dist.Normal(a, c.exp()))

        e = pyro.sample("e", dist.Normal(0, 1))
        f = pyro.sample("f", dist.Normal(0, 1))
        g = pyro.sample("g", dist.Bernoulli(logits=e + f),
                        obs=torch.tensor(0.))

        with pyro.plate("p", len(data)):
            d_ = d.detach()  # this results in a known failure
            h = pyro.sample("h", dist.Normal(c, d_.exp()))
            i = pyro.deterministic("i", h + 1)
            j = pyro.sample("j", dist.Delta(h + 1), obs=h + 1)
            k = pyro.sample("k", dist.Normal(a, j.exp()),
                            obs=data)

        return [a, b, c, d, e, f, g, h, i, j, k]

    data = torch.randn(3)
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
            "h": {"h": {"p"}, "c": _},  # [sic]
            "k": {"k": {"p"}, "a": _, "h": {"p"}},
        },
        "posterior_dependencies": {
            "a": {"a": _, "b": _, "c": _, "d": _, "h": _, "k": _},
            "b": {"b": _, "c": _},
            "c": {"c": _, "d": _, "h": _},  # [sic]
            "d": {"d": _},
            "e": {"e": _, "g": _, "f": _},
            "f": {"f": _, "g": _},
            "h": {"h": {"p"}, "k": {"p"}},
        },
    }
    assert actual == expected


def test_plate_coupling():
    # x  x
    #  ||
    #  y
    #
    # This results in posterior dependency structure:
    #
    #   x  x  y
    # x ?  ?  ?
    # x ?  ?  ?

    def model(data):
        with pyro.plate("p", len(data)):
            x = pyro.sample("x", dist.Normal(0, 1))
        pyro.sample("y", dist.Normal(x.sum(), 1),
                    obs=data.sum())

    data = torch.randn(2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": {"p"}},
            "y": {"y": set(), "x": set()},
        },
        "posterior_dependencies": {
            "x": {"x": set(), "y": set()},
        },
    }
    assert actual == expected


def test_plate_collider():
    # x x    y y
    #   \\  //
    #    zzzz
    #
    # This results in posterior dependency structure:
    #
    #   x x y y z z z z
    # x ?   ? ? ? ?
    # x   ? ? ?     ? ?
    # y     ?   ?   ?
    # y       ?   ?   ?

    def model(data):
        i_plate = pyro.plate("i", data.shape[0], dim=-2)
        j_plate = pyro.plate("j", data.shape[1], dim=-1)

        with i_plate:
            x = pyro.sample("x", dist.Normal(0, 1))
        with j_plate:
            y = pyro.sample("y", dist.Normal(0, 1))
        with i_plate, j_plate:
            pyro.sample("z", dist.Normal(x, y.exp()),
                        obs=data)

    data = torch.randn(3, 2)
    actual = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "x": {"x": {"i"}},
            "y": {"y": {"j"}},
            "z": {"x": {"i"}, "y": {"j"}, "z": {"i", "j"}},
        },
        "posterior_dependencies": {
            "x": {"x": {"i"}, "y": set(), "z": {"i"}},
            "y": {"y": {"j"}, "z": {"j"}},
        }
    }
    assert actual == expected


def test_plate_dependency():
    #    w                              w
    #      \  x1 x2      unroll    x1  / \  x2
    #       \  || y1 y2  =====>  y1 | /   \ | y2
    #        \ || //               \|/     \|/
    #         z1 z2                z1       z2
    #
    # This allows posterior dependency structure:
    #
    #    w x x y y z z
    #  w ? ? ? ? ? ? ?
    #  x   ?   ?   ?
    #  x     ?   ?   ?
    #  y       ?   ?
    #  y         ?   ?

    def model(data):
        w = pyro.sample("w", dist.Normal(0, 1))
        with pyro.plate("p", len(data)):
            x = pyro.sample("x", dist.Normal(0, 1))
            y = pyro.sample("y", dist.Normal(0, 1))
            pyro.sample("z", dist.Normal(w + x + y, 1),
                        obs=data)

    data = torch.rand(2)
    acutal = get_dependencies(model, (data,))
    expected = {
        "prior_dependencies": {
            "w": {"w": set()},
            "x": {"x": {"p"}},
            "y": {"y": {"p"}},
            "z": {"w": set(), "x": {"p"}, "y": {"p"}, "z": {"p"}},
        },
        "posterior_dependencies": {
            "w": {"w": set(), "x": set(), "y": set(), "z": set()},
            "x": {"x": {"p"}, "y": {"p"}, "z": {"p"}},
            "y": {"y": {"p"}, "z": {"p"}},
        },
    }
    assert acutal == expected
