# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import io
import logging
import pickle
import warnings
from unittest import TestCase

import pytest
import torch
import torch.nn as nn
from queue import Queue

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import Bernoulli, Categorical, Normal
from pyro.poutine.runtime import _DIM_ALLOCATOR, NonlocalExit
from pyro.poutine.util import all_escape, discrete_escape
from tests.common import assert_equal, assert_not_equal, assert_close

logger = logging.getLogger(__name__)


def eq(x, y, prec=1e-10):
    return (torch.norm(x - y).item() < prec)


# XXX name is a bit silly
class NormalNormalNormalHandlerTestCase(TestCase):

    def setUp(self):
        pyro.clear_param_store()

        def model():
            latent1 = pyro.sample("latent1",
                                  Normal(torch.zeros(2),
                                         torch.ones(2)))
            latent2 = pyro.sample("latent2",
                                  Normal(latent1,
                                         5 * torch.ones(2)))
            x_dist = Normal(latent2, torch.ones(2))
            pyro.sample("obs", x_dist, obs=torch.ones(2))
            return latent1

        def guide():
            loc1 = pyro.param("loc1", torch.randn(2, requires_grad=True))
            scale1 = pyro.param("scale1", torch.ones(2, requires_grad=True))
            pyro.sample("latent1", Normal(loc1, scale1))

            loc2 = pyro.param("loc2", torch.randn(2, requires_grad=True))
            scale2 = pyro.param("scale2", torch.ones(2, requires_grad=True))
            latent2 = pyro.sample("latent2", Normal(loc2, scale2))
            return latent2

        self.model = model
        self.guide = guide

        self.model_sites = ["latent1", "latent2",
                            "obs",
                            "_INPUT", "_RETURN"]

        self.guide_sites = ["latent1", "latent2",
                            "loc1", "scale1",
                            "loc2", "scale2",
                            "_INPUT", "_RETURN"]

        self.full_sample_sites = {"latent1": "latent1", "latent2": "latent2"}
        self.partial_sample_sites = {"latent1": "latent1"}


class TraceHandlerTests(NormalNormalNormalHandlerTestCase):

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

    def test_trace_param_only(self):
        model_trace = poutine.trace(self.model, param_only=True).get_trace()
        assert all(site["type"] == "param" for site in model_trace.nodes.values())


class ReplayHandlerTests(NormalNormalNormalHandlerTestCase):

    def test_replay_full(self):
        guide_trace = poutine.trace(self.guide).get_trace()
        model_trace = poutine.trace(poutine.replay(self.model, trace=guide_trace)).get_trace()
        for name in self.full_sample_sites.keys():
            assert_equal(model_trace.nodes[name]["value"],
                         guide_trace.nodes[name]["value"])

    def test_replay_full_repeat(self):
        model_trace = poutine.trace(self.model).get_trace()
        ftr = poutine.trace(poutine.replay(self.model, trace=model_trace))
        tr11 = ftr.get_trace()
        tr12 = ftr.get_trace()
        tr2 = poutine.trace(poutine.replay(self.model, trace=model_trace)).get_trace()
        for name in self.full_sample_sites.keys():
            assert_equal(tr11.nodes[name]["value"], tr12.nodes[name]["value"])
            assert_equal(tr11.nodes[name]["value"], tr2.nodes[name]["value"])
            assert_equal(model_trace.nodes[name]["value"], tr11.nodes[name]["value"])
            assert_equal(model_trace.nodes[name]["value"], tr2.nodes[name]["value"])


class BlockHandlerTests(NormalNormalNormalHandlerTestCase):

    def test_block_hide_fn(self):
        model_trace = poutine.trace(
            poutine.block(self.model,
                          hide_fn=lambda msg: "latent" in msg["name"],
                          expose=["latent1"])
        ).get_trace()
        assert "latent1" not in model_trace
        assert "latent2" not in model_trace
        assert "obs" in model_trace

    def test_block_expose_fn(self):
        model_trace = poutine.trace(
            poutine.block(self.model,
                          expose_fn=lambda msg: "latent" in msg["name"],
                          hide=["latent1"])
        ).get_trace()
        assert "latent1" in model_trace
        assert "latent2" in model_trace
        assert "obs" not in model_trace

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


class QueueHandlerDiscreteTest(TestCase):

    def setUp(self):

        # simple Gaussian-mixture HMM
        def model():
            probs = pyro.param("probs", torch.tensor([[0.8], [0.3]]))
            loc = pyro.param("loc", torch.tensor([[-0.1], [0.9]]))
            scale = torch.ones(1, 1)

            latents = [torch.ones(1)]
            observes = []
            for t in range(3):

                latents.append(
                    pyro.sample("latent_{}".format(str(t)),
                                Bernoulli(probs[latents[-1][0].long().data])))

                observes.append(
                    pyro.sample("observe_{}".format(str(t)),
                                Normal(loc[latents[-1][0].long().data], scale),
                                obs=torch.ones(1)))
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
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


class LiftHandlerTests(TestCase):

    def setUp(self):
        pyro.clear_param_store()

        def loc1_prior(tensor, *args, **kwargs):
            flat_tensor = tensor.view(-1)
            m = torch.zeros(flat_tensor.size(0))
            s = torch.ones(flat_tensor.size(0))
            return Normal(m, s).sample().view(tensor.size())

        def scale1_prior(tensor, *args, **kwargs):
            flat_tensor = tensor.view(-1)
            m = torch.zeros(flat_tensor.size(0))
            s = torch.ones(flat_tensor.size(0))
            return Normal(m, s).sample().view(tensor.size()).exp()

        def loc2_prior(tensor, *args, **kwargs):
            flat_tensor = tensor.view(-1)
            m = torch.zeros(flat_tensor.size(0))
            return Bernoulli(m).sample().view(tensor.size())

        def scale2_prior(tensor, *args, **kwargs):
            return scale1_prior(tensor)

        def bias_prior(tensor, *args, **kwargs):
            return loc2_prior(tensor)

        def weight_prior(tensor, *args, **kwargs):
            return scale1_prior(tensor)

        def stoch_fn(tensor, *args, **kwargs):
            loc = torch.zeros(tensor.size())
            scale = torch.ones(tensor.size())
            return pyro.sample("sample", Normal(loc, scale))

        def guide():
            loc1 = pyro.param("loc1", torch.randn(2, requires_grad=True))
            scale1 = pyro.param("scale1", torch.ones(2, requires_grad=True))
            pyro.sample("latent1", Normal(loc1, scale1))

            loc2 = pyro.param("loc2", torch.randn(2, requires_grad=True))
            scale2 = pyro.param("scale2", torch.ones(2, requires_grad=True))
            latent2 = pyro.sample("latent2", Normal(loc2, scale2))
            return latent2

        def dup_param_guide():
            a = pyro.param("loc")
            b = pyro.param("loc")
            assert a == b

        self.model = Model()
        self.guide = guide
        self.dup_param_guide = dup_param_guide
        self.prior = scale1_prior
        self.prior_dict = {"loc1": loc1_prior, "scale1": scale1_prior, "loc2": loc2_prior, "scale2": scale2_prior}
        self.partial_dict = {"loc1": loc1_prior, "scale1": scale1_prior}
        self.nn_prior = {"fc.bias": bias_prior, "fc.weight": weight_prior}
        self.fn = stoch_fn
        self.data = torch.randn(2, 2)

    def test_splice(self):
        tr = poutine.trace(self.guide).get_trace()
        lifted_tr = poutine.trace(poutine.lift(self.guide, prior=self.prior)).get_trace()
        for name in tr.nodes.keys():
            if name in ('loc1', 'loc2', 'scale1', 'scale2'):
                assert name not in lifted_tr
            else:
                assert name in lifted_tr

    def test_memoize(self):
        poutine.trace(poutine.lift(self.dup_param_guide, prior=dist.Normal(0, 1)))()

    def test_prior_dict(self):
        tr = poutine.trace(self.guide).get_trace()
        lifted_tr = poutine.trace(poutine.lift(self.guide, prior=self.prior_dict)).get_trace()
        for name in tr.nodes.keys():
            assert name in lifted_tr
            if name in {'scale1', 'loc1', 'scale2', 'loc2'}:
                assert name + "_prior" == lifted_tr.nodes[name]['fn'].__name__
            if tr.nodes[name]["type"] == "param":
                assert lifted_tr.nodes[name]["type"] == "sample"
                assert not lifted_tr.nodes[name]["is_observed"]

    def test_unlifted_param(self):
        tr = poutine.trace(self.guide).get_trace()
        lifted_tr = poutine.trace(poutine.lift(self.guide, prior=self.partial_dict)).get_trace()
        for name in tr.nodes.keys():
            assert name in lifted_tr
            if name in ('scale1', 'loc1'):
                assert name + "_prior" == lifted_tr.nodes[name]['fn'].__name__
                assert lifted_tr.nodes[name]["type"] == "sample"
                assert not lifted_tr.nodes[name]["is_observed"]
            if name in ('scale2', 'loc2'):
                assert lifted_tr.nodes[name]["type"] == "param"

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_random_module(self):
        pyro.clear_param_store()
        with pyro.validation_enabled():
            lifted_tr = poutine.trace(pyro.random_module("name", self.model, prior=self.prior)).get_trace()
        for name in lifted_tr.nodes.keys():
            if lifted_tr.nodes[name]["type"] == "param":
                assert lifted_tr.nodes[name]["type"] == "sample"
                assert not lifted_tr.nodes[name]["is_observed"]

    @pytest.mark.filterwarnings('ignore::FutureWarning')
    def test_random_module_warn(self):
        pyro.clear_param_store()
        bad_prior = {'foo': None}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pyro.validation_enabled():
                poutine.trace(pyro.random_module("name", self.model, prior=bad_prior)).get_trace()
            assert len(w), 'No warnings were raised'
            for warning in w:
                logger.info(warning)

    @pytest.mark.filterwarnings('ignore::FutureWarning')
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


class QueueHandlerMixedTest(TestCase):

    def setUp(self):

        # Simple model with 1 continuous + 1 discrete + 1 continuous variable.
        def model():
            p = torch.tensor([0.5])
            loc = torch.zeros(1)
            scale = torch.ones(1)

            x = pyro.sample("x", Normal(loc, scale))  # Before the discrete variable.
            y = pyro.sample("y", Bernoulli(p))
            z = pyro.sample("z", Normal(loc, scale))  # After the discrete variable.
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


class IndirectLambdaHandlerTests(TestCase):

    def setUp(self):

        def model(batch_size_outer=2, batch_size_inner=2):
            data = [[torch.ones(1)] * 2] * 2
            loc_latent = pyro.sample("loc_latent", dist.Normal(torch.zeros(1), torch.ones(1)))
            for i in pyro.plate("plate_outer", 2, batch_size_outer):
                for j in pyro.plate("plate_inner_%d" % i, 2, batch_size_inner):
                    pyro.sample("z_%d_%d" % (i, j), dist.Normal(loc_latent + data[i][j], torch.ones(1)))

        self.model = model
        self.expected_nodes = set(["z_0_0", "z_0_1", "z_1_0", "z_1_1", "loc_latent",
                                   "_INPUT", "_RETURN"])
        self.expected_edges = set([
            ("loc_latent", "z_0_0"), ("loc_latent", "z_0_1"),
            ("loc_latent", "z_1_0"), ("loc_latent", "z_1_1"),
        ])

    def test_graph_structure(self):
        tracegraph = poutine.trace(self.model, graph_type="dense").get_trace()
        # Ignore structure on plate_* nodes.
        actual_nodes = set(n for n in tracegraph.nodes if not n.startswith("plate_"))
        actual_edges = set((n1, n2) for n1, n2 in tracegraph.edges
                           if not n1.startswith("plate_") if not n2.startswith("plate_"))
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


class ConditionHandlerTests(NormalNormalNormalHandlerTestCase):

    def test_condition(self):
        data = {"latent2": torch.randn(2)}
        tr2 = poutine.trace(poutine.condition(self.model, data=data)).get_trace()
        assert "latent2" in tr2
        assert tr2.nodes["latent2"]["type"] == "sample" and \
            tr2.nodes["latent2"]["is_observed"]
        assert tr2.nodes["latent2"]["value"] is data["latent2"]

    def test_trace_data(self):
        tr1 = poutine.trace(
            poutine.block(self.model, expose_types=["sample"])).get_trace()
        tr2 = poutine.trace(
            poutine.condition(self.model, data=tr1)).get_trace()
        assert tr2.nodes["latent2"]["type"] == "sample" and \
            tr2.nodes["latent2"]["is_observed"]
        assert tr2.nodes["latent2"]["value"] is tr1.nodes["latent2"]["value"]

    def test_stack_overwrite_behavior(self):
        data1 = {"latent2": torch.randn(2)}
        data2 = {"latent2": torch.randn(2)}
        with poutine.trace() as tr:
            cm = poutine.condition(poutine.condition(self.model, data=data1),
                                   data=data2)
            cm()
        assert tr.trace.nodes['latent2']['value'] is data2['latent2']

    def test_stack_success(self):
        data1 = {"latent1": torch.randn(2)}
        data2 = {"latent2": torch.randn(2)}
        tr = poutine.trace(
            poutine.condition(poutine.condition(self.model, data=data1),
                              data=data2)).get_trace()
        assert tr.nodes["latent1"]["type"] == "sample" and \
            tr.nodes["latent1"]["is_observed"]
        assert tr.nodes["latent1"]["value"] is data1["latent1"]
        assert tr.nodes["latent2"]["type"] == "sample" and \
            tr.nodes["latent2"]["is_observed"]
        assert tr.nodes["latent2"]["value"] is data2["latent2"]


class UnconditionHandlerTests(NormalNormalNormalHandlerTestCase):

    def test_uncondition(self):
        unconditioned_model = poutine.uncondition(self.model)
        unconditioned_trace = poutine.trace(unconditioned_model).get_trace()
        conditioned_trace = poutine.trace(self.model).get_trace()
        assert_equal(conditioned_trace.nodes["obs"]["value"], torch.ones(2))
        assert_not_equal(unconditioned_trace.nodes["obs"]["value"], torch.ones(2))

    def test_undo_uncondition(self):
        unconditioned_model = poutine.uncondition(self.model)
        reconditioned_model = pyro.condition(unconditioned_model, {"obs": torch.ones(2)})
        reconditioned_trace = poutine.trace(reconditioned_model).get_trace()
        assert_equal(reconditioned_trace.nodes["obs"]["value"], torch.ones(2))


class EscapeHandlerTests(TestCase):

    def setUp(self):

        # Simple model with 1 continuous + 1 discrete + 1 continuous variable.
        def model():
            p = torch.tensor([0.5])
            loc = torch.zeros(1)
            scale = torch.ones(1)

            x = pyro.sample("x", Normal(loc, scale))  # Before the discrete variable.
            y = pyro.sample("y", Bernoulli(p))
            z = pyro.sample("z", Normal(loc, scale))  # After the discrete variable.
            return dict(x=x, y=y, z=z)

        self.sites = ["x", "y", "z", "_INPUT", "_RETURN"]
        self.model = model

    def test_discrete_escape(self):
        try:
            poutine.escape(self.model,
                           escape_fn=functools.partial(discrete_escape,
                                                       poutine.Trace()))()
            assert False
        except NonlocalExit as e:
            assert e.site["name"] == "y"

    def test_all_escape(self):
        try:
            poutine.escape(self.model,
                           escape_fn=functools.partial(all_escape,
                                                       poutine.Trace()))()
            assert False
        except NonlocalExit as e:
            assert e.site["name"] == "x"

    def test_trace_compose(self):
        tm = poutine.trace(self.model)
        try:
            poutine.escape(tm,
                           escape_fn=functools.partial(all_escape,
                                                       poutine.Trace()))()
            assert False
        except NonlocalExit:
            assert "x" in tm.trace
            try:
                tem = poutine.trace(
                    poutine.escape(self.model,
                                   escape_fn=functools.partial(all_escape,
                                                               poutine.Trace())))
                tem()
                assert False
            except NonlocalExit:
                assert "x" not in tem.trace


class InferConfigHandlerTests(TestCase):
    def setUp(self):
        def model():
            pyro.param("p", torch.zeros(1, requires_grad=True))
            pyro.sample("a", Bernoulli(torch.tensor([0.5])),
                        infer={"enumerate": "parallel"})
            pyro.sample("b", Bernoulli(torch.tensor([0.5])))

        self.model = model

        def config_fn(site):
            if site["type"] == "sample":
                return {"blah": True}
            else:
                return {}

        self.config_fn = config_fn

    def test_infer_config_sample(self):
        cfg_model = poutine.infer_config(self.model, config_fn=self.config_fn)

        tr = poutine.trace(cfg_model).get_trace()

        assert tr.nodes["a"]["infer"] == {"enumerate": "parallel", "blah": True}
        assert tr.nodes["b"]["infer"] == {"blah": True}
        assert tr.nodes["p"]["infer"] == {}


@pytest.mark.parametrize('first_available_dim', [-1, -2, -3])
@pytest.mark.parametrize('depth', [0, 1, 2])
def test_enumerate_poutine(depth, first_available_dim):
    num_particles = 2

    def model():
        pyro.sample("x", Bernoulli(0.5))
        for i in range(depth):
            pyro.sample("a_{}".format(i), Bernoulli(0.5), infer={"enumerate": "parallel"})

    model = poutine.enum(model, first_available_dim=first_available_dim)
    model = poutine.trace(model)

    for i in range(num_particles):
        tr = model.get_trace()
        tr.compute_log_prob()
        log_prob = sum(site["log_prob"] for name, site in tr.iter_stochastic_nodes())
        actual_shape = log_prob.shape
        expected_shape = (2,) * depth
        if depth:
            expected_shape = expected_shape + (1,) * (-1 - first_available_dim)
        assert actual_shape == expected_shape, 'error on iteration {}'.format(i)


@pytest.mark.parametrize('first_available_dim', [-1, -2, -3])
@pytest.mark.parametrize('depth', [0, 1, 2])
def test_replay_enumerate_poutine(depth, first_available_dim):
    num_particles = 2
    y_dist = Categorical(torch.tensor([0.5, 0.25, 0.25]))

    def guide():
        pyro.sample("y", y_dist, infer={"enumerate": "parallel"})

    guide = poutine.enum(guide, first_available_dim=first_available_dim - depth)
    guide = poutine.trace(guide)
    guide_trace = guide.get_trace()

    def model():
        pyro.sample("x", Bernoulli(0.5))
        for i in range(depth):
            pyro.sample("a_{}".format(i), Bernoulli(0.5), infer={"enumerate": "parallel"})
        pyro.sample("y", y_dist, infer={"enumerate": "parallel"})
        for i in range(depth):
            pyro.sample("b_{}".format(i), Bernoulli(0.5), infer={"enumerate": "parallel"})

    model = poutine.enum(model, first_available_dim=first_available_dim)
    model = poutine.replay(model, trace=guide_trace)
    model = poutine.trace(model)

    for i in range(num_particles):
        tr = model.get_trace()
        assert tr.nodes["y"]["value"] is guide_trace.nodes["y"]["value"]
        tr.compute_log_prob()
        log_prob = sum(site["log_prob"] for name, site in tr.iter_stochastic_nodes())
        actual_shape = log_prob.shape
        expected_shape = (2,) * depth + (3,) + (2,) * depth + (1,) * (-1 - first_available_dim)
        assert actual_shape == expected_shape, 'error on iteration {}'.format(i)


@pytest.mark.parametrize("has_rsample", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_plate_preserves_has_rsample(has_rsample, depth):
    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        with pyro.plate_stack("plates", (2,) * depth):
            return pyro.sample("x", dist.Normal(loc, 1).has_rsample_(has_rsample))

    x = guide()
    assert x.dim() == depth
    assert x.requires_grad == has_rsample


def test_plate_error_on_enter():
    def model():
        with pyro.plate('foo', 0):
            pass

    assert len(_DIM_ALLOCATOR._stack) == 0
    with pytest.raises(ZeroDivisionError):
        poutine.trace(model)()
    assert len(_DIM_ALLOCATOR._stack) == 0, 'stack was not cleaned on error'


def test_decorator_interface_primitives():

    @poutine.trace
    def model():
        pyro.param("p", torch.zeros(1, requires_grad=True))
        pyro.sample("a", Bernoulli(torch.tensor([0.5])),
                    infer={"enumerate": "parallel"})
        pyro.sample("b", Bernoulli(torch.tensor([0.5])))

    tr = model.get_trace()
    assert isinstance(tr, poutine.Trace)
    assert tr.graph_type == "flat"

    @poutine.trace(graph_type="dense")
    def model():
        pyro.param("p", torch.zeros(1, requires_grad=True))
        pyro.sample("a", Bernoulli(torch.tensor([0.5])),
                    infer={"enumerate": "parallel"})
        pyro.sample("b", Bernoulli(torch.tensor([0.5])))

    tr = model.get_trace()
    assert isinstance(tr, poutine.Trace)
    assert tr.graph_type == "dense"

    tr2 = poutine.trace(poutine.replay(model, trace=tr)).get_trace()

    assert_equal(tr2.nodes["a"]["value"], tr.nodes["a"]["value"])


def test_decorator_interface_queue():

    sites = ["x", "y", "z", "_INPUT", "_RETURN"]
    queue = Queue()
    queue.put(poutine.Trace())

    @poutine.queue(queue=queue)
    def model():
        p = torch.tensor([0.5])
        loc = torch.zeros(1)
        scale = torch.ones(1)

        x = pyro.sample("x", Normal(loc, scale))
        y = pyro.sample("y", Bernoulli(p))
        z = pyro.sample("z", Normal(loc, scale))
        return dict(x=x, y=y, z=z)

    tr = poutine.trace(model).get_trace()
    for name in sites:
        assert name in tr


def test_method_decorator_interface_condition():

    class cls_model:

        @poutine.condition(data={"b": torch.tensor(1.)})
        def model(self, p):
            self._model(p)

        def _model(self, p):
            pyro.sample("a", Bernoulli(p))
            pyro.sample("b", Bernoulli(torch.tensor([0.5])))

    tr = poutine.trace(cls_model().model).get_trace(0.5)
    assert isinstance(tr, poutine.Trace)
    assert tr.graph_type == "flat"
    assert tr.nodes["b"]["is_observed"] and tr.nodes["b"]["value"].item() == 1.


def test_trace_log_prob_err_msg():
    def model(v):
        pyro.sample("test_site", dist.Beta(1., 1.), obs=v)

    tr = poutine.trace(model).get_trace(torch.tensor(2.))
    exp_msg = r"Error while computing log_prob at site 'test_site':\s*" \
              r"The value argument must be within the support"
    with pytest.raises(ValueError, match=exp_msg):
        tr.compute_log_prob()


def test_trace_log_prob_sum_err_msg():
    def model(v):
        pyro.sample("test_site", dist.Beta(1., 1.), obs=v)

    tr = poutine.trace(model).get_trace(torch.tensor(2.))
    exp_msg = r"Error while computing log_prob_sum at site 'test_site':\s*" \
              r"The value argument must be within the support"
    with pytest.raises(ValueError, match=exp_msg):
        tr.log_prob_sum()


def test_trace_score_parts_err_msg():
    def guide(v):
        pyro.sample("test_site", dist.Beta(1., 1.), obs=v)

    tr = poutine.trace(guide).get_trace(torch.tensor(2.))
    exp_msg = r"Error while computing score_parts at site 'test_site':\s*" \
              r"The value argument must be within the support"
    with pytest.raises(ValueError, match=exp_msg):
        tr.compute_score_parts()


def _model(a=torch.tensor(1.), b=torch.tensor(1.)):
    latent = pyro.sample("latent", dist.Beta(a, b))
    return pyro.sample("test_site", dist.Bernoulli(latent), obs=torch.tensor(1))


@pytest.mark.parametrize('wrapper', [
    lambda fn: poutine.block(fn),
    lambda fn: poutine.condition(fn, {'latent': 0.9}),
    lambda fn: poutine.enum(fn, -1),
    lambda fn: poutine.replay(fn, poutine.trace(fn).get_trace()),
])
def test_pickling(wrapper):
    wrapped = wrapper(_model)
    buffer = io.BytesIO()
    # default protocol cannot serialize torch.Size objects (see https://github.com/pytorch/pytorch/issues/20823)
    torch.save(wrapped, buffer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    buffer.seek(0)
    deserialized = torch.load(buffer)
    obs = torch.tensor(0.5)
    pyro.set_rng_seed(0)
    actual_trace = poutine.trace(deserialized).get_trace(obs)
    pyro.set_rng_seed(0)
    expected_trace = poutine.trace(wrapped).get_trace(obs)
    assert tuple(actual_trace) == tuple(expected_trace.nodes)
    assert_close([actual_trace.nodes[site]['value'] for site in actual_trace.stochastic_nodes],
                 [expected_trace.nodes[site]['value'] for site in expected_trace.stochastic_nodes])


def test_arg_kwarg_error():

    def model():
        pyro.param("p", torch.zeros(1, requires_grad=True))
        pyro.sample("a", Bernoulli(torch.tensor([0.5])),
                    infer={"enumerate": "parallel"})
        pyro.sample("b", Bernoulli(torch.tensor([0.5])))

    with pytest.raises(ValueError, match="not callable"):
        with poutine.mask(False):
            model()

    with poutine.mask(mask=False):
        model()
