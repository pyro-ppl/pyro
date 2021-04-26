# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.testing.fakes import NonreparameterizedNormal
from pyro.infer.inspect import get_dependencies


def test_get_dependencies_simple():
    data = torch.randn(3)

    def model(data):
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", NonreparameterizedNormal(a, 0))
        c = pyro.sample("c", dist.Normal(b, 1))
        d = pyro.sample("d", dist.Normal(a, c.exp()))
        with pyro.plate("data", len(data)):
            e = pyro.sample("e", dist.Normal(c, d.detach().exp()))
            pyro.sample("obs", dist.Normal(a, e.exp()), obs=data)

    actual = get_dependencies(model, (data,))
    expected = {
        "a": [],
        "b": ["a"],
        "c": ["b"],
        "d": ["a", "c"],
        "e": ["c"],
    }
    assert actual == expected
