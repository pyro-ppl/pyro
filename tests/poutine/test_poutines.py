import torch
from six.moves.queue import Queue
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.distributions import DiagNormal, Bernoulli
import pyro.distributions as dist
from tests.common import TestCase, assert_equal
from pyro.util import ng_ones, ng_zeros


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
        guide_trace = poutine.trace(self.guide).get_trace()
        model_trace = poutine.trace(self.model).get_trace()
        for name in model_trace.keys():
            assert name in self.model_sites

        for name in guide_trace.keys():
            assert name in self.guide_sites
            assert guide_trace[name]["type"] != "observe"

    def test_trace_return(self):
        model_trace = poutine.trace(self.model).get_trace()
        assert_equal(model_trace["latent1"]["value"], model_trace["_RETURN"]["value"])


class ReplayPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_replay_full(self):
        guide_trace = poutine.trace(self.guide).get_trace()
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace()
        for name in self.full_sample_sites.keys():
            assert_equal(model_trace[name]["value"], guide_trace[name]["value"])

    def test_replay_partial(self):
        guide_trace = poutine.trace(self.guide).get_trace()
        model_trace = poutine.trace(poutine.replay(self.model,
                                                   guide_trace,
                                                   sites=self.partial_sample_sites)).get_trace()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                assert_equal(model_trace[name]["value"], guide_trace[name]["value"])
            else:
                assert not eq(model_trace[name]["value"],
                              guide_trace[name]["value"])

    def test_replay_full_repeat(self):
        model_trace = poutine.trace(self.model).get_trace()
        ftr = poutine.trace(poutine.replay(self.model, model_trace))
        tr11 = ftr.get_trace()
        tr12 = ftr.get_trace()
        tr2 = poutine.trace(poutine.replay(self.model, model_trace)).get_trace()
        for name in self.full_sample_sites.keys():
            assert_equal(tr11[name]["value"], tr12[name]["value"])
            assert_equal(tr11[name]["value"], tr2[name]["value"])
            assert_equal(model_trace[name]["value"], tr11[name]["value"])
            assert_equal(model_trace[name]["value"], tr2[name]["value"])


class BlockPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_block_full(self):
        model_trace = poutine.trace(poutine.block(self.model)).get_trace()
        guide_trace = poutine.trace(poutine.block(self.guide)).get_trace()
        for name in model_trace.keys():
            assert model_trace[name]["type"] in ("args", "return")
        for name in guide_trace.keys():
            assert guide_trace[name]["type"] in ("args", "return")

    def test_block_full_hide(self):
        model_trace = poutine.trace(poutine.block(self.model,
                                                  hide=self.model_sites)).get_trace()
        guide_trace = poutine.trace(poutine.block(self.guide,
                                                  hide=self.guide_sites)).get_trace()
        for name in model_trace.keys():
            assert model_trace[name]["type"] in ("args", "return")
        for name in guide_trace.keys():
            assert guide_trace[name]["type"] in ("args", "return")

    def test_block_full_expose(self):
        model_trace = poutine.trace(poutine.block(self.model,
                                                  expose=self.model_sites)).get_trace()
        guide_trace = poutine.trace(poutine.block(self.guide,
                                                  expose=self.guide_sites)).get_trace()
        for name in self.model_sites:
            assert name in model_trace
        for name in self.guide_sites:
            assert name in guide_trace

    def test_block_full_hide_expose(self):
        try:
            poutine.block(self.model,
                          hide=self.partial_sample_sites.keys(),
                          expose=self.partial_sample_sites.keys())()
            assert False
        except AssertionError:
            assert True

    def test_block_partial_hide(self):
        model_trace = poutine.trace(
            poutine.block(self.model, hide=self.partial_sample_sites.keys())).get_trace()
        guide_trace = poutine.trace(
            poutine.block(self.guide, hide=self.partial_sample_sites.keys())).get_trace()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                name not in model_trace
                name not in guide_trace
            else:
                name in model_trace
                name in guide_trace

    def test_block_partial_expose(self):
        model_trace = poutine.trace(
            poutine.block(self.model, expose=self.partial_sample_sites.keys())).get_trace()
        guide_trace = poutine.trace(
            poutine.block(self.guide, expose=self.partial_sample_sites.keys())).get_trace()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                assert name in model_trace
                assert name in guide_trace
            else:
                name not in model_trace
                name not in guide_trace


class QueuePoutineDiscreteTest(TestCase):

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
        tr = f.get_trace()
        for name in self.sites:
            assert name in tr

    def test_queue_enumerate(self):
        f = poutine.trace(poutine.queue(self.model, queue=self.queue))
        trs = []
        while not self.queue.empty():
            trs.append(f.get_trace())
        assert len(trs) == 2 ** 3

        true_latents = set()
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    true_latents.add((i1, i2, i3))

        tr_latents = set()
        for tr in trs:
            tr_latents.add(tuple([tr[name]["value"].view(-1).data[0] for name in tr
                                  if tr[name]["type"] == "sample"]))

        assert true_latents == tr_latents

    def test_queue_max_tries(self):
        f = poutine.queue(self.model, queue=self.queue, max_tries=3)
        try:
            f()
            assert False
        except ValueError:
            assert True


class QueuePoutineMixedTest(TestCase):

    def setUp(self):

        # Simple model with 1 continuous + 1 discrete + 1 continuous variable.
        def model():
            p = Variable(torch.Tensor([0.5]))
            mu = Variable(torch.zeros(1))
            sigma = Variable(torch.ones(1))

            x = pyro.sample("x", DiagNormal(mu, sigma))  # Before the discrete variable.
            y = pyro.sample("y", Bernoulli(p))
            z = pyro.sample("z", DiagNormal(mu, sigma))  # After the discrete variable.
            return dict(x=x, y=y, z=z)

        self.sites = ["x", "y", "z", "_INPUT", "_RETURN"]
        self.model = model
        self.queue = Queue()
        self.queue.put(poutine.Trace())

    def test_queue_single(self):
        f = poutine.trace(poutine.queue(self.model, queue=self.queue))
        tr = f.get_trace()
        for name in self.sites:
            assert name in tr

    def test_queue_enumerate(self):
        f = poutine.trace(poutine.queue(self.model, queue=self.queue))
        trs = []
        while not self.queue.empty():
            trs.append(f.get_trace())
        assert len(trs) == 2

        values = [
            {name: tr[name]['value'].view(-1).data[0] for name in tr.keys()
             if tr[name]['type'] == 'sample'}
            for tr in trs
        ]

        expected_ys = set([0, 1])
        actual_ys = set([value["y"] for value in values])
        assert actual_ys == expected_ys

        # Check that x was sampled the same on all each paths.
        assert values[0]["x"] == values[1]["x"]

        # Check that y was sampled differently on each path.
        assert values[0]["z"] != values[1]["z"]  # Almost surely true.


class IndirectLambdaPoutineTests(TestCase):

    def setUp(self):

        def model(batch_size_outer=2, batch_size_inner=2):
            mu_latent = pyro.sample("mu_latent", dist.diagnormal, ng_zeros(1), ng_ones(1))

            def outer(i, x):
                pyro.map_data("map_inner_%d" % i, x, lambda _i, _x:
                              inner(i, _i, _x), batch_size=batch_size_inner)

            def inner(i, _i, _x):
                pyro.sample("z_%d_%d" % (i, _i), dist.diagnormal, mu_latent + _x, ng_ones(1))

            pyro.map_data("map_outer", [[ng_ones(1)] * 2] * 2, lambda i, x:
                          outer(i, x), batch_size=batch_size_outer)

            return mu_latent

        self.model = model
        self.expected_nodes = set(['z_0_0', 'z_0_1', 'z_1_0', 'z_1_1', 'mu_latent'])
        self.expected_edges = set([('mu_latent', 'z_0_0'), ('mu_latent', 'z_0_1'),
                                   ('mu_latent', 'z_1_0'), ('mu_latent', 'z_1_1')])

    def test_graph_structure(self):
        tracegraph = poutine.tracegraph(self.model).get_trace()
        assert set(tracegraph.get_graph().nodes()) == self.expected_nodes
        assert set(tracegraph.get_graph().edges()) == self.expected_edges

    def test_scale_factors(self):
        def _test_scale_factor(batch_size_outer, batch_size_inner, expected):
            trace = poutine.tracegraph(self.model).get_trace(batch_size_outer=batch_size_outer,
                                                             batch_size_inner=batch_size_inner).get_trace()
            scale_factors = []
            for node in ['z_0_0', 'z_0_1', 'z_1_0', 'z_1_1']:
                if node in trace:
                    scale_factors.append(trace[node]['scale'])
            assert scale_factors == expected

        _test_scale_factor(1, 1, [4.0])
        _test_scale_factor(2, 2, [1.0] * 4)
        _test_scale_factor(1, 2, [2.0] * 2)
        _test_scale_factor(2, 1, [2.0] * 2)


class ConditionPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_condition(self):
        data = {"latent2": Variable(torch.randn(2))}
        tr2 = poutine.trace(poutine.condition(self.model, data=data)).get_trace()
        self.assertTrue("latent2" in tr2)
        self.assertTrue(tr2["latent2"]["type"] == "observe")
        self.assertTrue(eq(tr2["latent2"]["value"], data["latent2"]))

    def test_do(self):
        data = {"latent2": Variable(torch.randn(2))}
        tr3 = poutine.trace(poutine.do(self.model, data=data)).get_trace()
        self.assertTrue("latent2" not in tr3)

    def test_trace_data(self):
        tr1 = poutine.trace(
            poutine.block(self.model, expose_types=["sample"])).get_trace()
        tr2 = poutine.trace(
            poutine.condition(self.model, data=tr1)).get_trace()
        assert tr2["latent2"]["type"] == "observe"
        assert eq(tr2["latent2"]["value"], tr1["latent2"]["value"])

    def test_stack_overwrite_failure(self):
        data1 = {"latent2": Variable(torch.randn(2))}
        data2 = {"latent2": Variable(torch.randn(2))}
        try:
            poutine.condition(poutine.condition(self.model, data=data1),
                              data=data2)()
            assert False
        except AssertionError:
            assert True

    def test_stack_success(self):
        data1 = {"latent1": Variable(torch.randn(2))}
        data2 = {"latent2": Variable(torch.randn(2))}
        tr = poutine.trace(
            poutine.condition(poutine.condition(self.model, data=data1),
                              data=data2)).get_trace()
        assert tr["latent1"]["type"] == "observe"
        assert eq(tr["latent1"]["value"], data1["latent1"])
        assert tr["latent2"]["type"] == "observe"
        assert eq(tr["latent2"]["value"], data2["latent2"])
