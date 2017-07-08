import numpy as np
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal, Bernoulli
import pyro.poutine as poutine
from pyro.util import memoize

from tests.common import TestCase


def eq(x, y, prec=1e-10):
    return (torch.norm(x - y).data[0] < prec)


class SimplePoutineTests(TestCase):

    def setUp(self):

        def model():
            latent = pyro.sample("latent",
                                 DiagNormal(Variable(torch.zeros(1)),
                                            5 * Variable(torch.ones(1))))
            x_dist = DiagNormal(latent, Variable(torch.ones(1)))
            x = pyro.observe("obs", x_dist, Variable(torch.ones(1)))
            return latent

        def guide():
            latent = pyro.sample("latent",
                                 DiagNormal(Variable(torch.zeros(1)),
                                            5 * Variable(torch.ones(1))))
            # x_dist = DiagNormal(latent, Variable(torch.ones(1)))
            return latent

        self.model = model
        self.guide = guide

    def test_trace(self):
        tr = poutine.trace(self.model)()

        # check that everything is there

        # check that latent == _RETURN

    def test_replay(self):

        # test full replay

        # test partial replay
        pass

    def test_cache(self):

        # test full replay

        # test partial replay
        pass

    def test_memoize(self):

        # test with a single argument

        # test with multiple non-kw arguments

        # test with multiple non-kw and kw arguments

        # test with tensors
        pass

    def test_block(self):

        # test full block

        # test partial block with hide

        # test partial block with expose

        # test partial block with hide and expose
        pass

    def test_queue(self):

        # test single trace

        # test full enumeration

        # test max_tries failure case

        # test queue exhaustion failure case
        pass


class CompoundPoutineTests(TestCase):

    def setUp(self):
        pass

    def test_trace_replay(self):
        """
        Make sure trace and replay interact correctly
        """
        model_trace = poutine.trace(self.model)()
        guide_trace = poutine.trace(self.guide)()

        model_trace_replay = poutine.replay(poutine.trace(self.model), guide_trace)()
        model_replay_trace = poutine.trace(poutine.replay(self.model, guide_trace))()
        model_replay_ret = poutine.replay(self.model, guide_trace)()

        self.assertTrue(eq(model_trace_replay["_RETURN"]["value"], model_replay_ret))

        self.assertTrue(eq(model_replay_ret, guide_trace["latent"]["value"]))

        self.assertTrue(eq(model_replay_trace["latent"]["value"],
                           guide_trace["latent"]["value"]))

        self.assertFalse(eq(model_replay_trace["latent"]["value"],
                            model_trace_replay["latent"]["value"]))

        self.assertFalse(eq(model_trace["latent"]["value"],
                            guide_trace["latent"]["value"]))

    def test_block_trace_replay(self):
        pass
