# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.infer.inspect import get_dependencies


def test_get_dependencies():
    data = torch.randn(3)

    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", NonreparameterizedNormal(a, 0))
        c = pyro.sample("c", dist.Normal(b, 1))
        d = pyro.sample("d", dist.Normal(a, c.exp()))

        e = pyro.sample("e", dist.Normal(0, 1))
        f = pyro.sample("f", dist.Normal(0, 1))
        g = pyro.sample("g", dist.Bernoulli(logits=e + f),
                        obs=torch.tensor(0.))

        with pyro.plate("data", len(data)):
            d_ = d.detach()  # this results in a known failure
            h = pyro.sample("h", dist.Normal(c, d_.exp()))
            i = pyro.deterministic("i", h + 1)
            j = pyro.sample("j", dist.Delta(h + 1), obs=h + 1)
            k = pyro.sample("k", dist.Normal(a, j.exp()),
                            obs=data)

        return [a, b, c, d, e, f, g, h, i, j, k]

    actual = get_dependencies(model, (data,))
    expected = {
        "a": [],
        "b": ["a"],
        "c": ["a", "b"],
        "d": ["a", "c"],
        "e": [],
        "f": ["e"],
        "h": ["a", "c"],
    }
    assert actual == expected
