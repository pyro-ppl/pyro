import numpy as np
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
import pyro.poutine as poutine

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

        self.model = model

        def guide():
            latent = pyro.sample("latent",
                                 DiagNormal(Variable(torch.zeros(1)),
                                            5 * Variable(torch.ones(1))))
            # x_dist = DiagNormal(latent, Variable(torch.ones(1)))
            return latent

        self.guide = guide

    def test_trace_replay(self):
        """
        some simple invariants on a single example, but woefully incomplete
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
