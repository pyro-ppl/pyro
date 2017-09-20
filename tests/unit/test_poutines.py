import torch
from six.moves.queue import Queue
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.distributions import DiagNormal, Bernoulli
from tests.common import TestCase


def eq(x, y, prec=1e-10):
    return (torch.norm(x - y).data[0] < prec)


# XXX name is a bit silly
class NormalNormalNormalPoutineTestCase(TestCase):

    def setUp(self):
        pyro.get_param_store().clear()

        def model():
            latent1 = pyro.sample("latent1",
                                  DiagNormal(Variable(torch.zeros(2)),
                                             Variable(torch.ones(2))))
            latent2 = pyro.sample("latent2",
                                  DiagNormal(latent1,
                                             5 * Variable(torch.ones(2))))
            x_dist = DiagNormal(latent2, Variable(torch.ones(2)))
            pyro.observe("obs", x_dist, Variable(torch.ones(2)))
            return latent1

        def guide():
            mu1 = pyro.param("mu1", Variable(torch.randn(2), requires_grad=True))
            sigma1 = pyro.param("sigma1", Variable(torch.ones(2), requires_grad=True))
            pyro.sample("latent1", DiagNormal(mu1, sigma1))

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

    def test_replay_full_repeat(self):
        model_trace = poutine.trace(self.model)()
        ftr = poutine.trace(poutine.replay(self.model, model_trace))
        tr11 = ftr()
        tr12 = ftr()
        tr2 = poutine.trace(poutine.replay(self.model, model_trace))()
        for name in self.full_sample_sites.keys():
            self.assertTrue(eq(tr11[name]["value"], tr12[name]["value"]))
            self.assertTrue(eq(tr11[name]["value"], tr2[name]["value"]))
            self.assertTrue(eq(model_trace[name]["value"], tr11[name]["value"]))
            self.assertTrue(eq(model_trace[name]["value"], tr2[name]["value"]))


class CachePoutineTests(NormalNormalNormalPoutineTestCase):

    def test_cache_full(self):
        cached_model = poutine.trace(poutine.cache(self.model))
        model_trace_1 = cached_model()
        model_trace_2 = cached_model()
        for name in self.full_sample_sites.keys():
            self.assertTrue(eq(model_trace_1[name]["value"],
                               model_trace_2[name]["value"]))

    def test_cache_partial(self):
        cached_model = poutine.trace(
            poutine.cache(self.model, sites=self.partial_sample_sites))
        model_trace_1 = cached_model()
        model_trace_2 = cached_model()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                self.assertTrue(eq(model_trace_1[name]["value"],
                                   model_trace_2[name]["value"]))
            else:
                self.assertFalse(eq(model_trace_1[name]["value"],
                                    model_trace_2[name]["value"]))


class BlockPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_block_full(self):
        model_trace = poutine.trace(poutine.block(self.model))()
        guide_trace = poutine.trace(poutine.block(self.guide))()
        for name in model_trace.keys():
            self.assertTrue(model_trace[name]["type"] in ("args", "return"))
        for name in guide_trace.keys():
            self.assertTrue(guide_trace[name]["type"] in ("args", "return"))

    def test_block_full_hide(self):
        model_trace = poutine.trace(poutine.block(self.model,
                                                  hide=self.model_sites))()
        guide_trace = poutine.trace(poutine.block(self.guide,
                                                  hide=self.guide_sites))()
        for name in model_trace.keys():
            self.assertTrue(model_trace[name]["type"] in ("args", "return"))
        for name in guide_trace.keys():
            self.assertTrue(guide_trace[name]["type"] in ("args", "return"))

    def test_block_full_expose(self):
        model_trace = poutine.trace(poutine.block(self.model,
                                                  expose=self.model_sites))()
        guide_trace = poutine.trace(poutine.block(self.guide,
                                                  expose=self.guide_sites))()
        for name in self.model_sites:
            self.assertTrue(name in model_trace)
        for name in self.guide_sites:
            self.assertTrue(name in guide_trace)

    def test_block_full_hide_expose(self):
        try:
            poutine.block(self.model,
                          hide=self.partial_sample_sites.keys(),
                          expose=self.partial_sample_sites.keys())()
            self.assertTrue(False)
        except AssertionError:
            self.assertTrue(True)

    def test_block_partial_hide(self):
        model_trace = poutine.trace(
            poutine.block(self.model, hide=self.partial_sample_sites.keys()))()
        guide_trace = poutine.trace(
            poutine.block(self.guide, hide=self.partial_sample_sites.keys()))()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                self.assertFalse(name in model_trace)
                self.assertFalse(name in guide_trace)
            else:
                self.assertTrue(name in model_trace)
                self.assertTrue(name in guide_trace)

    def test_block_partial_expose(self):
        model_trace = poutine.trace(
            poutine.block(self.model, expose=self.partial_sample_sites.keys()))()
        guide_trace = poutine.trace(
            poutine.block(self.guide, expose=self.partial_sample_sites.keys()))()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                self.assertTrue(name in model_trace)
                self.assertTrue(name in guide_trace)
            else:
                self.assertFalse(name in model_trace)
                self.assertFalse(name in guide_trace)


class QueuePoutineTests(TestCase):

    def setUp(self):

        # simple Gaussian-mixture HMM
        def model():
            ps = pyro.param("ps", Variable(torch.Tensor([[0.8], [0.3]])))
            mu = pyro.param("mu", Variable(torch.Tensor([[-0.1], [0.9]])))
            sigma = Variable(torch.ones(1))

            latents = [Variable(torch.ones(1))]
            observes = []
            for t in range(3):

                latents.append(
                    pyro.sample("latent_{}".format(str(t)),
                                Bernoulli(ps[latents[-1][0].long().data])))

                observes.append(
                    pyro.observe("observe_{}".format(str(t)),
                                 DiagNormal(mu[latents[-1][0].long().data], sigma),
                                 pyro.ones(1)))
            return latents

        self.sites = ["observe_{}".format(str(t)) for t in range(3)] + \
                     ["latent_{}".format(str(t)) for t in range(3)] + \
                     ["_INPUT", "_RETURN"]
        self.model = model
        self.queue = Queue()
        self.queue.put(poutine.Trace())

    def test_queue_single(self):
        f = poutine.trace(poutine.queue(self.model, queue=self.queue))
        tr = f()
        for name in self.sites:
            self.assertTrue(name in tr)

    def test_queue_enumerate(self):
        f = poutine.trace(poutine.queue(self.model, queue=self.queue))
        trs = []
        while not self.queue.empty():
            trs.append(f())
        self.assertTrue(len(trs) == 2 ** 3)

        true_latents = set()
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    true_latents.add((i1, i2, i3))

        tr_latents = set()
        for tr in trs:
            tr_latents.add(tuple([tr[name]["value"].view(-1).data[0] for name in tr
                                  if tr[name]["type"] == "sample"]))

        self.assertTrue(true_latents == tr_latents)

    def test_queue_max_tries(self):
        f = poutine.queue(self.model, queue=self.queue, max_tries=3)
        try:
            f()
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)


class ConditionPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_condition(self):
        data = {"latent2": Variable(torch.randn(2))}
        tr1 = poutine.trace(self.model)()
        tr2 = poutine.trace(poutine.condition(self.model, data=data))()
        tr3 = poutine.trace(poutine.do(self.model, data=data))()

        self.assertTrue("latent2" in tr2)
        self.assertTrue(tr2["latent2"]["type"] == "observe")
        self.assertTrue(eq(tr2["latent2"]["value"], data["latent2"]))

        self.assertTrue("latent2" not in tr3)
