import pytest

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.contrib.autoname import glom_name


class B(object):
    def __init__(self):
        b = 3


def test_dynamic_simple():

    @glom_name
    def model():

        x = pyro.sample(dist.Bernoulli(0.5))
        y = pyro.sample(dist.Bernoulli(0.5))
        z = [x, y, None]
        i = 2
        z[i] = pyro.sample(dist.Bernoulli(0.5))
        zz = {}
        zz["a"] = pyro.sample(dist.Bernoulli(0.5))
        ab = B()
        ab.b = pyro.sample(dist.Bernoulli(0.5))
        zz["c"] = [B(), None, None, None]
        zz["c"][0].b = pyro.sample(dist.Bernoulli(0.5))
        for j in range(1, 3):
            zz["c"][j] = pyro.sample(dist.Bernoulli(0.5))

    expected_names = [
        "x",
        "y",
        "z[2]",
        "zz['a']",
        "ab.b",
        "zz['c'][0].b",
        "zz['c'][1]",
        "zz['c'][2]",
    ]

    tr = poutine.trace(model).get_trace()
    actual_names = [k for k, v in tr.nodes.items() if v["type"] == "sample"]
    print(tr.nodes)

    assert actual_names == expected_names
