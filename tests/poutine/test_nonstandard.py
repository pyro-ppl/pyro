from __future__ import absolute_import, division, print_function

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

import pyro.poutine.nonstandard as nonstandard


def test_provenance():

    @nonstandard.make_nonstandard
    def add(x, y):
        return x + y

    @nonstandard.make_nonstandard
    def mul(x, y):
        return x * y

    @nonstandard.ProvenanceMessenger()
    def blah():
        return add(mul(2, 3), 4)

    b = blah()
    assert isinstance(b, nonstandard.ProvenanceBox)
    print(b.value)
