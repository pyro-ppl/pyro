from __future__ import absolute_import, division, print_function

import functools
from unittest import TestCase

import pytest
import torch
import torch.nn as nn
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import Bernoulli, Normal
from pyro.poutine.util import all_escape, discrete_escape, NonlocalExit
from pyro.util import ng_ones, ng_zeros
from six.moves.queue import Queue
from tests.common import assert_equal


def eq(x, y, prec=1e-10):
    return (torch.norm(x - y).item() < prec)


# XXX name is a bit silly
class NormalNormalNormalPoutineTestCase(TestCase):

    def setUp(self):
        pyro.clear_param_store()

        def model():
            latent1 = pyro.sample("latent1",
                                  Normal(Variable(torch.zeros(2)),
                                         Variable(torch.ones(2))))
            latent2 = pyro.sample("latent2",
                                  Normal(latent1,
                                         5 * Variable(torch.ones(2))))
            x_dist = Normal(latent2, Variable(torch.ones(2)))
            pyro.observe("obs", x_dist, Variable(torch.ones(2)))
            return latent1

        def guide():
            mu1 = pyro.param("mu1", Variable(torch.randn(2), requires_grad=True))
            sigma1 = pyro.param("sigma1", Variable(torch.ones(2), requires_grad=True))
            pyro.sample("latent1", Normal(mu1, sigma1))

            mu2 = pyro.param("mu2", Variable(torch.randn(2), requires_grad=True))
            sigma2 = pyro.param("sigma2", Variable(torch.ones(2), requires_grad=True))
            latent2 = pyro.sample("latent2", Normal(mu2, sigma2))
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
        for name in model_trace.nodes.keys():
            assert name in self.model_sites

        for name in guide_trace.nodes.keys():
            assert name in self.guide_sites
            assert guide_trace.nodes[name]["type"] in \
                ("args", "return", "sample", "param")
            if guide_trace.nodes[name]["type"] == "sample":
                assert not guide_trace.nodes[name]["is_observed"]

    def test_trace_return(self):
        model_trace = poutine.trace(self.model).get_trace()
        assert_equal(model_trace.nodes["latent1"]["value"],
                     model_trace.nodes["_RETURN"]["value"])


class ReplayPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_replay_full(self):
        guide_trace = poutine.trace(self.guide).get_trace()
        model_trace = poutine.trace(poutine.replay(self.model, guide_trace)).get_trace()
        for name in self.full_sample_sites.keys():
            assert_equal(model_trace.nodes[name]["value"],
                         guide_trace.nodes[name]["value"])

    def test_replay_partial(self):
        guide_trace = poutine.trace(self.guide).get_trace()
        model_trace = poutine.trace(poutine.replay(self.model,
                                                   guide_trace,
                                                   sites=self.partial_sample_sites)).get_trace()
        for name in self.full_sample_sites.keys():
            if name in self.partial_sample_sites:
                assert_equal(model_trace.nodes[name]["value"],
                             guide_trace.nodes[name]["value"])
            else:
                assert not eq(model_trace.nodes[name]["value"],
                              guide_trace.nodes[name]["value"])

    def test_replay_full_repeat(self):
        model_trace = poutine.trace(self.model).get_trace()
        ftr = poutine.trace(poutine.replay(self.model, model_trace))
        tr11 = ftr.get_trace()
        tr12 = ftr.get_trace()
        tr2 = poutine.trace(poutine.replay(self.model, model_trace)).get_trace()
        for name in self.full_sample_sites.keys():
            assert_equal(tr11.nodes[name]["value"], tr12.nodes[name]["value"])
            assert_equal(tr11.nodes[name]["value"], tr2.nodes[name]["value"])
            assert_equal(model_trace.nodes[name]["value"], tr11.nodes[name]["value"])
            assert_equal(model_trace.nodes[name]["value"], tr2.nodes[name]["value"])


class BlockPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_block_full(self):
        model_trace = poutine.trace(poutine.block(self.model)).get_trace()
        guide_trace = poutine.trace(poutine.block(self.guide)).get_trace()
        for name in model_trace.nodes.keys():
            assert model_trace.nodes[name]["type"] in ("args", "return")
        for name in guide_trace.nodes.keys():
            assert guide_trace.nodes[name]["type"] in ("args", "return")

    def test_block_full_hide(self):
        model_trace = poutine.trace(poutine.block(self.model,
                                                  hide=self.model_sites)).get_trace()
        guide_trace = poutine.trace(poutine.block(self.guide,
                                                  hide=self.guide_sites)).get_trace()
        for name in model_trace.nodes.keys():
            assert model_trace.nodes[name]["type"] in ("args", "return")
        for name in guide_trace.nodes.keys():
            assert guide_trace.nodes[name]["type"] in ("args", "return")

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
                assert name not in model_trace
                assert name not in guide_trace
            else:
                assert name in model_trace
                assert name in guide_trace

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
                assert name not in model_trace
                assert name not in guide_trace

    def test_block_tutorial_case(self):
        model_trace = poutine.trace(self.model).get_trace()
        guide_trace = poutine.trace(
            poutine.block(self.guide, hide_types=["observe"])).get_trace()

        assert "latent1" in model_trace
        assert "latent1" in guide_trace
        assert "obs" in model_trace
        assert "obs" not in guide_trace


class QueuePoutineDiscreteTest(TestCase):

    def setUp(self):

        # simple Gaussian-mixture HMM
        def model():
            ps = pyro.param("ps", Variable(torch.Tensor([[0.8], [0.3]])))
            mu = pyro.param("mu", Variable(torch.Tensor([[-0.1], [0.9]])))
            sigma = Variable(torch.ones(1, 1))

            latents = [Variable(torch.ones(1))]
            observes = []
            for t in range(3):

                latents.append(
                    pyro.sample("latent_{}".format(str(t)),
                                Bernoulli(ps[latents[-1][0].long().data])))

                observes.append(
                    pyro.observe("observe_{}".format(str(t)),
                                 Normal(mu[latents[-1][0].long().data], sigma),
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

        tr_latents = []
        for tr in trs:
            tr_latents.append(tuple([int(tr.nodes[name]["value"].view(-1).item()) for name in tr
                                     if tr.nodes[name]["type"] == "sample" and
                                     not tr.nodes[name]["is_observed"]]))

        assert true_latents == set(tr_latents)

    def test_queue_max_tries(self):
        f = poutine.queue(self.model, queue=self.queue, max_tries=3)
        with pytest.raises(ValueError):
            f()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


class LiftPoutineTests(TestCase):

    def setUp(self):
        pyro.clear_param_store()

        def mu1_prior(tensor, *args, **kwargs):
            flat_tensor = tensor.view(-1)
            m = Variable(torch.zeros(flat_tensor.size(0)))
            s = Variable(torch.ones(flat_tensor.size(0)))
            return Normal(m, s).sample().view(tensor.size())

        def sigma1_prior(tensor, *args, **kwargs):
            flat_tensor = tensor.view(-1)
            m = Variable(torch.zeros(flat_tensor.size(0)))
            s = Variable(torch.ones(flat_tensor.size(0)))
            return Normal(m, s).sample().view(tensor.size())

        def mu2_prior(tensor, *args, **kwargs):
            flat_tensor = tensor.view(-1)
            m = Variable(torch.zeros(flat_tensor.size(0)))
            return Bernoulli(m).sample().view(tensor.size())

        def sigma2_prior(tensor, *args, **kwargs):
            return sigma1_prior(tensor)

        def bias_prior(tensor, *args, **kwargs):
            return mu2_prior(tensor)

        def weight_prior(tensor, *args, **kwargs):
            return sigma1_prior(tensor)

        def stoch_fn(tensor, *args, **kwargs):
            mu = Variable(torch.zeros(tensor.size()))
            sigma = Variable(torch.ones(tensor.size()))
            return pyro.sample("sample", Normal(mu, sigma))

        def guide():
            mu1 = pyro.param("mu1", Variable(torch.randn(2), requires_grad=True))
            sigma1 = pyro.param("sigma1", Variable(torch.ones(2), requires_grad=True))
            pyro.sample("latent1", Normal(mu1, sigma1))

            mu2 = pyro.param("mu2", Variable(torch.randn(2), requires_grad=True))
            sigma2 = pyro.param("sigma2", Variable(torch.ones(2), requires_grad=True))
            latent2 = pyro.sample("latent2", Normal(mu2, sigma2))
            return latent2

        self.model = Model()
        self.guide = guide
        self.prior = mu1_prior
        self.prior_dict = {"mu1": mu1_prior, "sigma1": sigma1_prior, "mu2": mu2_prior, "sigma2": sigma2_prior}
        self.partial_dict = {"mu1": mu1_prior, "sigma1": sigma1_prior}
        self.nn_prior = {"fc.bias": bias_prior, "fc.weight": weight_prior}
        self.fn = stoch_fn
        self.data = Variable(torch.randn(2, 2))

    def test_splice(self):
        tr = poutine.trace(self.guide).get_trace()
        lifted_tr = poutine.trace(poutine.lift(self.guide, prior=self.prior)).get_trace()
        for name in tr.nodes.keys():
            if name in ('mu1', 'mu2', 'sigma1', 'sigma2'):
                assert name not in lifted_tr
            else:
                assert name in lifted_tr

    def test_prior_dict(self):
        tr = poutine.trace(self.guide).get_trace()
        lifted_tr = poutine.trace(poutine.lift(self.guide, prior=self.prior_dict)).get_trace()
        for name in tr.nodes.keys():
            assert name in lifted_tr
            if name in {'sigma1', 'mu1', 'sigma2', 'mu2'}:
                assert name + "_prior" == lifted_tr.nodes[name]['fn'].__name__
            if tr.nodes[name]["type"] == "param":
                assert lifted_tr.nodes[name]["type"] == "sample"
                assert not lifted_tr.nodes[name]["is_observed"]

    def test_unlifted_param(self):
        tr = poutine.trace(self.guide).get_trace()
        lifted_tr = poutine.trace(poutine.lift(self.guide, prior=self.partial_dict)).get_trace()
        for name in tr.nodes.keys():
            assert name in lifted_tr
            if name in ('sigma1', 'mu1'):
                assert name + "_prior" == lifted_tr.nodes[name]['fn'].__name__
                assert lifted_tr.nodes[name]["type"] == "sample"
                assert not lifted_tr.nodes[name]["is_observed"]
            if name in ('sigma2', 'mu2'):
                assert lifted_tr.nodes[name]["type"] == "param"

    def test_random_module(self):
        pyro.clear_param_store()
        lifted_tr = poutine.trace(pyro.random_module("name", self.model, prior=self.prior)).get_trace()
        for name in lifted_tr.nodes.keys():
            if lifted_tr.nodes[name]["type"] == "param":
                assert lifted_tr.nodes[name]["type"] == "sample"
                assert not lifted_tr.nodes[name]["is_observed"]

    def test_random_module_prior_dict(self):
        pyro.clear_param_store()
        lifted_nn = pyro.random_module("name", self.model, prior=self.nn_prior)
        lifted_tr = poutine.trace(lifted_nn).get_trace()
        for key_name in lifted_tr.nodes.keys():
            name = pyro.params.user_param_name(key_name)
            if name in {'fc.weight', 'fc.prior'}:
                dist_name = name[3:]
                assert dist_name + "_prior" == lifted_tr.nodes[key_name]['fn'].__name__
                assert lifted_tr.nodes[key_name]["type"] == "sample"
                assert not lifted_tr.nodes[key_name]["is_observed"]


class QueuePoutineMixedTest(TestCase):

    def setUp(self):

        # Simple model with 1 continuous + 1 discrete + 1 continuous variable.
        def model():
            p = Variable(torch.Tensor([0.5]))
            mu = Variable(torch.zeros(1))
            sigma = Variable(torch.ones(1))

            x = pyro.sample("x", Normal(mu, sigma))  # Before the discrete variable.
            y = pyro.sample("y", Bernoulli(p))
            z = pyro.sample("z", Normal(mu, sigma))  # After the discrete variable.
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
            {name: tr.nodes[name]['value'].view(-1).item() for name in tr.nodes.keys()
             if tr.nodes[name]['type'] == 'sample'}
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
            data = [[ng_ones(1)] * 2] * 2
            mu_latent = pyro.sample("mu_latent", dist.Normal(ng_zeros(1), ng_ones(1)))
            for i in pyro.irange("irange_outer", 2, batch_size_outer):
                for j in pyro.irange("irange_inner_%d" % i, 2, batch_size_inner):
                    pyro.sample("z_%d_%d" % (i, j), dist.Normal(mu_latent + data[i][j], ng_ones(1)))

        self.model = model
        self.expected_nodes = set(["z_0_0", "z_0_1", "z_1_0", "z_1_1", "mu_latent",
                                   "_INPUT", "_RETURN"])
        self.expected_edges = set([
            ("mu_latent", "z_0_0"), ("mu_latent", "z_0_1"),
            ("mu_latent", "z_1_0"), ("mu_latent", "z_1_1"),
        ])

    def test_graph_structure(self):
        tracegraph = poutine.trace(self.model, graph_type="dense").get_trace()
        # Ignore structure on irange_* nodes.
        actual_nodes = set(n for n in tracegraph.nodes() if not n.startswith("irange_"))
        actual_edges = set((n1, n2) for n1, n2 in tracegraph.edges
                           if not n1.startswith("irange_") if not n2.startswith("irange_"))
        assert actual_nodes == self.expected_nodes
        assert actual_edges == self.expected_edges

    def test_scale_factors(self):
        def _test_scale_factor(batch_size_outer, batch_size_inner, expected):
            trace = poutine.trace(self.model, graph_type="dense").get_trace(batch_size_outer=batch_size_outer,
                                                                            batch_size_inner=batch_size_inner)
            scale_factors = []
            for node in ['z_0_0', 'z_0_1', 'z_1_0', 'z_1_1']:
                if node in trace:
                    scale_factors.append(trace.nodes[node]['scale'])
            assert scale_factors == expected

        _test_scale_factor(1, 1, [4.0])
        _test_scale_factor(2, 2, [1.0] * 4)
        _test_scale_factor(1, 2, [2.0] * 2)
        _test_scale_factor(2, 1, [2.0] * 2)


class ConditionPoutineTests(NormalNormalNormalPoutineTestCase):

    def test_condition(self):
        data = {"latent2": Variable(torch.randn(2))}
        tr2 = poutine.trace(poutine.condition(self.model, data=data)).get_trace()
        assert "latent2" in tr2
        assert tr2.nodes["latent2"]["type"] == "sample" and \
            tr2.nodes["latent2"]["is_observed"]
        assert tr2.nodes["latent2"]["value"] is data["latent2"]

    def test_do(self):
        data = {"latent2": Variable(torch.randn(2))}
        tr3 = poutine.trace(poutine.do(self.model, data=data)).get_trace()
        assert "latent2" not in tr3

    def test_trace_data(self):
        tr1 = poutine.trace(
            poutine.block(self.model, expose_types=["sample"])).get_trace()
        tr2 = poutine.trace(
            poutine.condition(self.model, data=tr1)).get_trace()
        assert tr2.nodes["latent2"]["type"] == "sample" and \
            tr2.nodes["latent2"]["is_observed"]
        assert tr2.nodes["latent2"]["value"] is tr1.nodes["latent2"]["value"]

    def test_stack_overwrite_failure(self):
        data1 = {"latent2": Variable(torch.randn(2))}
        data2 = {"latent2": Variable(torch.randn(2))}
        cm = poutine.condition(poutine.condition(self.model, data=data1),
                               data=data2)
        with pytest.raises(AssertionError):
            cm()

    def test_stack_success(self):
        data1 = {"latent1": Variable(torch.randn(2))}
        data2 = {"latent2": Variable(torch.randn(2))}
        tr = poutine.trace(
            poutine.condition(poutine.condition(self.model, data=data1),
                              data=data2)).get_trace()
        assert tr.nodes["latent1"]["type"] == "sample" and \
            tr.nodes["latent1"]["is_observed"]
        assert tr.nodes["latent1"]["value"] is data1["latent1"]
        assert tr.nodes["latent2"]["type"] == "sample" and \
            tr.nodes["latent2"]["is_observed"]
        assert tr.nodes["latent2"]["value"] is data2["latent2"]

    def test_do_propagation(self):
        pyro.clear_param_store()

        def model():
            z = pyro.sample("z", Normal(10.0 * ng_ones(1), 0.0001 * ng_ones(1)))
            latent_prob = torch.exp(z) / (torch.exp(z) + ng_ones(1))
            flip = pyro.sample("flip", Bernoulli(latent_prob))
            return flip

        sample_from_model = model()
        z_data = {"z": -10.0 * ng_ones(1)}
        # under model flip = 1 with high probability; so do indirect DO surgery to make flip = 0
        sample_from_do_model = poutine.trace(poutine.do(model, data=z_data))()

        assert eq(sample_from_model, ng_ones(1))
        assert eq(sample_from_do_model, ng_zeros(1))


class EscapePoutineTests(TestCase):

    def setUp(self):

        # Simple model with 1 continuous + 1 discrete + 1 continuous variable.
        def model():
            p = Variable(torch.Tensor([0.5]))
            mu = Variable(torch.zeros(1))
            sigma = Variable(torch.ones(1))

            x = pyro.sample("x", Normal(mu, sigma))  # Before the discrete variable.
            y = pyro.sample("y", Bernoulli(p))
            z = pyro.sample("z", Normal(mu, sigma))  # After the discrete variable.
            return dict(x=x, y=y, z=z)

        self.sites = ["x", "y", "z", "_INPUT", "_RETURN"]
        self.model = model

    def test_discrete_escape(self):
        try:
            poutine.escape(self.model, functools.partial(discrete_escape,
                                                         poutine.Trace()))()
            assert False
        except NonlocalExit as e:
            assert e.site["name"] == "y"

    def test_all_escape(self):
        try:
            poutine.escape(self.model, functools.partial(all_escape,
                                                         poutine.Trace()))()
            assert False
        except NonlocalExit as e:
            assert e.site["name"] == "x"

    def test_trace_compose(self):
        tm = poutine.trace(self.model)
        try:
            poutine.escape(tm, functools.partial(all_escape, poutine.Trace()))()
            assert False
        except NonlocalExit:
            assert "x" in tm.trace
            try:
                tem = poutine.trace(
                    poutine.escape(self.model, functools.partial(all_escape,
                                                                 poutine.Trace())))
                tem()
                assert False
            except NonlocalExit:
                assert "x" not in tem.trace


class InferConfigPoutineTests(TestCase):
    def setUp(self):
        def model():
            pyro.param("p", Variable(torch.zeros(1), requires_grad=True))
            pyro.sample("a", Bernoulli(Variable(torch.Tensor([0.5]))),
                        infer={"enumerate": "parallel"})
            pyro.sample("b", Bernoulli(Variable(torch.Tensor([0.5]))))

        self.model = model

        def config_fn(site):
            if site["type"] == "sample":
                return {"blah": True}
            else:
                return {}

        self.config_fn = config_fn

    def test_infer_config_sample(self):
        cfg_model = poutine.infer_config(self.model, self.config_fn)

        tr = poutine.trace(cfg_model).get_trace()

        assert tr.nodes["a"]["infer"] == {"enumerate": "parallel", "blah": True}
        assert tr.nodes["b"]["infer"] == {"blah": True}
        assert tr.nodes["p"]["infer"] == {}
