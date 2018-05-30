import pytest

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.contrib.autoname.glom_named import sample, dynamic_scope


def test_dynamic_simple():

    def submodel():
        y = sample(dist.Bernoulli(0.5))
        return y

    class B(object):
        def __init__(self):
            b = 3

    @dynamic_scope
    def model():
        x = sample(dist.Bernoulli(0.5))
        y = sample(dist.Bernoulli(0.5))
        # xx = submodel()
        z = [x, y, None, None]
        i = 3
        z[i] = sample(dist.Bernoulli(0.5))
        zz = {}
        zz["a"] = sample(dist.Bernoulli(0.5))
        ab = B()
        ab.b = sample(dist.Bernoulli(0.5))
        # yy = submodel()
        # xyz = [sample(dist.Bernoulli(0.5)) for i in range(3)]
        # z[-1] = submodel()
        # for i in range(3):
        #     zi = sample(dist.Bernoulli(0.5))
        #     z.append(zi)
        return z

    tr = poutine.trace(model).get_trace()
    print(tr.nodes)
