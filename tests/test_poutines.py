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


# XXX name is a bit silly
class NormalNormalNormalPoutineTestCase(TestCase):

    def setUp(self):
        pyro._param_store._clear_cache()

        def model():
            latent1 = pyro.sample("latent1",
                                  DiagNormal(Variable(torch.zeros(2)),
                                             Variable(torch.ones(2))))
            latent2 = pyro.sample("latent2",
                                  DiagNormal(latent1,
                                             5 * Variable(torch.ones(2))))
            x_dist = DiagNormal(latent2, Variable(torch.ones(2)))
            x = pyro.observe("obs", x_dist, Variable(torch.ones(2)))
            return latent1

        def guide():
            mu1 = pyro.param("mu1", Variable(torch.randn(2), requires_grad=True))
            sigma1 = pyro.param("sigma1", Variable(torch.ones(2), requires_grad=True))
            latent1 = pyro.sample("latent1", DiagNormal(mu1, sigma1))

            mu2 = pyro.param("mu2", Variable(torch.randn(2), requires_grad=True))
            sigma2 = pyro.param("sigma2", Variable(torch.ones(2), requires_grad=True))
            latent2 = pyro.sample("latent2", DiagNormal(mu2, sigma2))
            return latent2

        self.model = model
        self.guide = guide

        self.model_sites = ["latent1", "latent2",
                            "obs",
                            "_INPUT", "_RETURN"]

        self.guide_sites = ["latent1", "latent2",
                            "mu1", "sigma1",
                            "mu2", "sigma2",
                            "_INPUT", "_RETURN"]

        self.full_sample_sites = {"latent1": "latent1", "latent2": "latent2"}
        self.partial_sample_sites = {"latent1": "latent1"}

        
class TracePoutineTests(NormalNormalNormalPoutineTestCase):
        
    def test_trace_full(self):
        guide_trace = poutine.trace(self.guide)()
        model_trace = poutine.trace(self.model)()
        for name in model_trace.keys():
            self.assertTrue(name in self.model_sites)

        for name in guide_trace.keys():
            self.assertTrue(name in self.guide_sites)
            self.assertFalse(guide_trace[name]["type"] == "observe")

    def test_trace_return(self):
        model_trace = poutine.trace(self.model)()
        self.assertTrue(eq(model_trace["latent1"]["value"],
                           model_trace["_RETURN"]["value"]))


class ReplayPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_replay_full(self):
        guide_trace = poutine.trace(self.guide)()
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace))()
        for name in self.full_sample_sites.keys():
            self.assertTrue(eq(model_trace[name]["value"],
                               guide_trace[name]["value"]))

    def test_replay_partial(self):
        guide_trace = poutine.trace(self.guide)()
        model_trace = poutine.trace(poutine.replay(self.model,
                                                   guide_trace,
                                                   sites=self.partial_sample_sites))()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                self.assertTrue(eq(model_trace[name]["value"],
                                   guide_trace[name]["value"]))
            else:
                self.assertFalse(eq(model_trace[name]["value"],
                                    guide_trace[name]["value"]))


class CachePoutineTests(NormalNormalNormalPoutineTestCase):
    
    def test_cache_full(self):
        cached_model = poutine.cache(poutine.trace(self.model))
        model_trace_1 = cached_model()
        model_trace_2 = cached_model()
        for name in self.full_sample_sites.keys():
            print(name, model_trace_1[name]["value"], model_trace_2[name]["value"])
            self.assertTrue(eq(model_trace_1[name]["value"],
                               model_trace_2[name]["value"]))

    def test_cache_partial(self):
        cached_model = poutine.cache(poutine.trace(self.model),
                                     sites=self.partial_sample_sites)
        model_trace_1 = cached_model()
        model_trace_2 = cached_model()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                self.assertTrue(eq(model_trace_1[name]["value"],
                                   model_trace_2[name]["value"]))
            else:
                self.assertFalse(eq(model_trace_1[name]["value"],
                                    model_trace_2[name]["value"]))


# class MemoizePoutineTests(NormalNormalNormalPoutineTestCase):
#     
#     def test_memoize(self):
# 
#         # test with a single argument
# 
#         # test with multiple non-kw arguments
# 
#         # test with multiple non-kw and kw arguments
# 
#         # test with tensors
#         pass


# class BlockPoutineTests(NormalNormalNormalPoutineTestCase):
#     
#     def setUp(self):
#         super(BlockPoutineTests, self).setUp()
#         # XXX set hide and expose sites?
#         
#     def test_block_full(self):
#         pass
# 
#     def test_block_full_hide(self):
#         pass
# 
#     def test_block_full_expose(self):
#         pass
# 
#     def test_block_full_hide_expose(self):
#         pass
#     
#     def test_block_partial_hide(self):
#         pass
# 
#     def test_block_partial_expose(self):
#         pass
# 
#     def test_block_partial_hide_expose(self):
#         pass
# 
# 
# class QueuePoutineTests(TestCase):
# 
#     def setUp(self):
#         pass
#     
#     def test_queue(self):
# 
#         # test single trace
# 
#         # test full enumeration
# 
#         # test max_tries failure case
# 
#         # test queue exhaustion failure case
#         pass


# class CompoundPoutineTests(NormalNormalNormalPoutineTestCase):
# 
#     def test_trace_replay(self):
#         """
#         Make sure trace and replay interact correctly
#         """
#         model_trace = poutine.trace(self.model)()
#         guide_trace = poutine.trace(self.guide)()
# 
#         model_trace_replay = poutine.replay(poutine.trace(self.model), guide_trace)()
#         model_replay_trace = poutine.trace(poutine.replay(self.model, guide_trace))()
#         model_replay_ret = poutine.replay(self.model, guide_trace)()
# 
#         self.assertTrue(eq(model_trace_replay["_RETURN"]["value"], model_replay_ret))
# 
#         self.assertTrue(eq(model_replay_ret, guide_trace["latent"]["value"]))
# 
#         self.assertTrue(eq(model_replay_trace["latent"]["value"],
#                            guide_trace["latent"]["value"]))
# 
#         self.assertFalse(eq(model_replay_trace["latent"]["value"],
#                             model_trace_replay["latent"]["value"]))
# 
#         self.assertFalse(eq(model_trace["latent"]["value"],
#                             guide_trace["latent"]["value"]))
# 
#     def test_block_trace_replay(self):
#         pass
