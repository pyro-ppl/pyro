# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import functools
import io
import warnings
from operator import attrgetter

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, Predictive
from pyro.infer.autoguide import (AutoCallable, AutoDelta, AutoDiagonalNormal, AutoDiscreteParallel, AutoGuide,
                                  AutoGuideList, AutoIAFNormal, AutoLaplaceApproximation, AutoLowRankMultivariateNormal,
                                  AutoNormal, AutoMultivariateNormal, init_to_feasible, init_to_mean, init_to_median,
                                  init_to_sample)
from pyro.nn.module import PyroModule, PyroParam, PyroSample
from pyro.optim import Adam
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match
from tests.common import assert_close, assert_equal


@pytest.mark.parametrize("auto_class", [
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
])
def test_scores(auto_class):
    def model():
        if auto_class is AutoIAFNormal:
            pyro.sample("z", dist.Normal(0.0, 1.0).expand([10]))
        else:
            pyro.sample("z", dist.Normal(0.0, 1.0))

    guide = auto_class(model)
    guide_trace = poutine.trace(guide).get_trace()
    model_trace = poutine.trace(poutine.replay(model, guide_trace)).get_trace()

    guide_trace.compute_log_prob()
    model_trace.compute_log_prob()

    prefix = auto_class.__name__
    if prefix != 'AutoNormal':
        assert '_{}_latent'.format(prefix) not in model_trace.nodes
        assert guide_trace.nodes['_{}_latent'.format(prefix)]['log_prob_sum'].item() != 0.0
    assert model_trace.nodes['z']['log_prob_sum'].item() != 0.0
    assert guide_trace.nodes['z']['log_prob_sum'].item() == 0.0


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
])
def test_factor(auto_class, Elbo):

    def model(log_factor):
        pyro.sample("z1", dist.Normal(0.0, 1.0))
        pyro.factor("f1", log_factor)
        pyro.sample("z2", dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1))
        with pyro.plate("plate", 3):
            pyro.factor("f2", log_factor)
            pyro.sample("z3", dist.Normal(torch.zeros(3), torch.ones(3)))

    guide = auto_class(model)
    elbo = Elbo(strict_enumeration_warning=False)
    elbo.loss(model, guide, torch.tensor(0.))  # initialize param store

    pyro.set_rng_seed(123)
    loss_5 = elbo.loss(model, guide, torch.tensor(5.))
    pyro.set_rng_seed(123)
    loss_4 = elbo.loss(model, guide, torch.tensor(4.))
    assert_close(loss_5 - loss_4, -1 - 3)


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
@pytest.mark.parametrize("init_loc_fn", [
    init_to_feasible,
    init_to_mean,
    init_to_median,
    init_to_sample,
])
@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_shapes(auto_class, init_loc_fn, Elbo):

    def model():
        pyro.sample("z1", dist.Normal(0.0, 1.0))
        pyro.sample("z2", dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1))
        with pyro.plate("plate", 3):
            pyro.sample("z3", dist.Normal(torch.zeros(3), torch.ones(3)))
        pyro.sample("z4", dist.MultivariateNormal(torch.zeros(2), torch.eye(2)))
        pyro.sample("z5", dist.Dirichlet(torch.ones(3)))
        pyro.sample("z6", dist.Normal(0, 1).expand((2,)).mask(torch.arange(2) > 0).to_event(1))

    guide = auto_class(model, init_loc_fn=init_loc_fn)
    elbo = Elbo(strict_enumeration_warning=False)
    loss = elbo.loss(model, guide)
    assert np.isfinite(loss), loss


@pytest.mark.xfail(reason="sequential plate is not yet supported")
@pytest.mark.parametrize('auto_class', [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO])
def test_iplate_smoke(auto_class, Elbo):

    def model():
        x = pyro.sample("x", dist.Normal(0, 1))
        assert x.shape == ()

        for i in pyro.plate("plate", 3):
            y = pyro.sample("y_{}".format(i), dist.Normal(0, 1).expand_by([2, 1 + i, 2]).to_event(3))
            assert y.shape == (2, 1 + i, 2)

        z = pyro.sample("z", dist.Normal(0, 1).expand_by([2]).to_event(1))
        assert z.shape == (2,)

        pyro.sample("obs", dist.Bernoulli(0.1), obs=torch.tensor(0))

    guide = auto_class(model)
    infer = SVI(model, guide, Adam({"lr": 1e-6}), Elbo(strict_enumeration_warning=False))
    infer.step()


def auto_guide_list_x(model):
    guide = AutoGuideList(model)
    guide.append(AutoDelta(poutine.block(model, expose=["x"])))
    guide.append(AutoDiagonalNormal(poutine.block(model, hide=["x"])))
    return guide


def auto_guide_callable(model):
    def guide_x():
        x_loc = pyro.param("x_loc", torch.tensor(1.))
        x_scale = pyro.param("x_scale", torch.tensor(.1), constraint=constraints.positive)
        pyro.sample("x", dist.Normal(x_loc, x_scale))

    def median_x():
        return {"x": pyro.param("x_loc", torch.tensor(1.))}

    guide = AutoGuideList(model)
    guide.append(AutoCallable(model, guide_x, median_x))
    guide.append(AutoDiagonalNormal(poutine.block(model, hide=["x"])))
    return guide


def auto_guide_module_callable(model):
    class GuideX(AutoGuide):
        def __init__(self, model):
            super().__init__(model)
            self.x_loc = nn.Parameter(torch.tensor(1.))
            self.x_scale = PyroParam(torch.tensor(.1), constraint=constraints.positive)

        def forward(self, *args, **kwargs):
            return {"x": pyro.sample("x", dist.Normal(self.x_loc, self.x_scale))}

        def median(self, *args, **kwargs):
            return {"x": self.x_loc.detach()}

    guide = AutoGuideList(model)
    guide.custom = GuideX(model)
    guide.diagnorm = AutoDiagonalNormal(poutine.block(model, hide=["x"]))
    return guide


def nested_auto_guide_callable(model):
    guide = AutoGuideList(model)
    guide.append(AutoDelta(poutine.block(model, expose=['x'])))
    guide_y = AutoGuideList(poutine.block(model, expose=['y']))
    guide_y.z = AutoIAFNormal(poutine.block(model, expose=['y']))
    guide.append(guide_y)
    return guide


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
    auto_guide_list_x,
    auto_guide_callable,
    auto_guide_module_callable,
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_feasible),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_mean),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_median),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_sample),
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_median(auto_class, Elbo):

    def model():
        pyro.sample("x", dist.Normal(0.0, 1.0))
        pyro.sample("y", dist.LogNormal(0.0, 1.0))
        pyro.sample("z", dist.Beta(2.0, 2.0))

    guide = auto_class(model)
    optim = Adam({'lr': 0.05, 'betas': (0.8, 0.99)})
    elbo = Elbo(strict_enumeration_warning=False,
                num_particles=100, vectorize_particles=True)
    infer = SVI(model, guide, optim, elbo)
    for _ in range(100):
        infer.step()

    if auto_class is AutoLaplaceApproximation:
        guide = guide.laplace_approximation()

    median = guide.median()
    assert_equal(median["x"], torch.tensor(0.0), prec=0.1)
    if auto_class is AutoDelta:
        assert_equal(median["y"], torch.tensor(-1.0).exp(), prec=0.1)
    else:
        assert_equal(median["y"], torch.tensor(1.0), prec=0.1)
    assert_equal(median["z"], torch.tensor(0.5), prec=0.1)


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
    auto_guide_list_x,
    auto_guide_module_callable,
    nested_auto_guide_callable,
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_feasible),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_mean),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_median),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_sample),
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_autoguide_serialization(auto_class, Elbo):
    def model():
        pyro.sample("x", dist.Normal(0.0, 1.0))
        with pyro.plate("plate", 2):
            pyro.sample("y", dist.LogNormal(0.0, 1.0))
            pyro.sample("z", dist.Beta(2.0, 2.0))
    guide = auto_class(model)
    guide()
    if auto_class is AutoLaplaceApproximation:
        guide = guide.laplace_approximation()
    pyro.set_rng_seed(0)
    expected = guide.call()
    names = sorted(guide())

    # Ignore tracer warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        # XXX: check_trace=True fails for AutoLaplaceApproximation
        traced_guide = torch.jit.trace_module(guide, {"call": ()}, check_trace=False)
    f = io.BytesIO()
    torch.jit.save(traced_guide, f)
    f.seek(0)
    guide_deser = torch.jit.load(f)

    # Check .call() result.
    pyro.set_rng_seed(0)
    actual = guide_deser.call()
    assert len(actual) == len(expected)
    for name, a, e in zip(names, actual, expected):
        assert_equal(a, e, msg="{}: {} vs {}".format(name, a, e))

    # Check named_parameters.
    expected_names = {name for name, _ in guide.named_parameters()}
    actual_names = {name for name, _ in guide_deser.named_parameters()}
    assert actual_names == expected_names
    for name in actual_names:
        # Get nested attributes.
        attr_get = attrgetter(name)
        assert_equal(attr_get(guide_deser), attr_get(guide).data)


@pytest.mark.parametrize("auto_class", [
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_quantiles(auto_class, Elbo):

    def model():
        pyro.sample("x", dist.Normal(0.0, 1.0))
        pyro.sample("y", dist.LogNormal(0.0, 1.0))
        pyro.sample("z", dist.Beta(2.0, 2.0))

    guide = auto_class(model)
    optim = Adam({'lr': 0.05, 'betas': (0.8, 0.99)})
    elbo = Elbo(strict_enumeration_warning=False,
                num_particles=100, vectorize_particles=True)
    infer = SVI(model, guide, optim, elbo)
    for _ in range(100):
        infer.step()

    if auto_class is AutoLaplaceApproximation:
        guide = guide.laplace_approximation()

    quantiles = guide.quantiles([0.1, 0.5, 0.9])
    median = guide.median()
    for name in ["x", "y", "z"]:
        assert_equal(median[name], quantiles[name][1])
    quantiles = {name: [v.item() for v in value] for name, value in quantiles.items()}

    assert -3.0 < quantiles["x"][0]
    assert quantiles["x"][0] + 1.0 < quantiles["x"][1]
    assert quantiles["x"][1] + 1.0 < quantiles["x"][2]
    assert quantiles["x"][2] < 3.0

    assert 0.01 < quantiles["y"][0]
    assert quantiles["y"][0] * 2.0 < quantiles["y"][1]
    assert quantiles["y"][1] * 2.0 < quantiles["y"][2]
    assert quantiles["y"][2] < 100.0

    assert 0.01 < quantiles["z"][0]
    assert quantiles["z"][0] + 0.1 < quantiles["z"][1]
    assert quantiles["z"][1] + 0.1 < quantiles["z"][2]
    assert quantiles["z"][2] < 0.99


@pytest.mark.parametrize("continuous_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
])
def test_discrete_parallel(continuous_class):
    K = 2
    data = torch.tensor([0., 1., 10., 11., 12.])

    def model(data):
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
        locs = pyro.sample('locs', dist.Normal(0, 10).expand_by([K]).to_event(1))
        scale = pyro.sample('scale', dist.LogNormal(0, 1))

        with pyro.plate('data', len(data)):
            weights = weights.expand(torch.Size((len(data),)) + weights.shape)
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

    guide = AutoGuideList(model)
    guide.append(continuous_class(poutine.block(model, hide=["assignment"])))
    guide.append(AutoDiscreteParallel(poutine.block(model, expose=["assignment"])))

    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    loss = elbo.loss_and_grads(model, guide, data)
    assert np.isfinite(loss), loss


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
])
def test_guide_list(auto_class):

    def model():
        pyro.sample("x", dist.Normal(0., 1.).expand([2]))
        pyro.sample("y", dist.MultivariateNormal(torch.zeros(5), torch.eye(5, 5)))

    guide = AutoGuideList(model)
    guide.append(auto_class(poutine.block(model, expose=["x"])))
    guide.append(auto_class(poutine.block(model, expose=["y"])))
    guide()


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
])
def test_callable(auto_class):

    def model():
        pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.MultivariateNormal(torch.zeros(5), torch.eye(5, 5)))

    def guide_x():
        x_loc = pyro.param("x_loc", torch.tensor(0.))
        pyro.sample("x", dist.Delta(x_loc))

    guide = AutoGuideList(model)
    guide.append(guide_x)
    guide.append(auto_class(poutine.block(model, expose=["y"])))
    values = guide()
    assert set(values) == set(["y"])


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
])
def test_callable_return_dict(auto_class):

    def model():
        pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.MultivariateNormal(torch.zeros(5), torch.eye(5, 5)))

    def guide_x():
        x_loc = pyro.param("x_loc", torch.tensor(0.))
        x = pyro.sample("x", dist.Delta(x_loc))
        return {"x": x}

    guide = AutoGuideList(model)
    guide.append(guide_x)
    guide.append(auto_class(poutine.block(model, expose=["y"])))
    values = guide()
    assert set(values) == set(["x", "y"])


def test_empty_model_error():
    def model():
        pass
    guide = AutoDiagonalNormal(model)
    with pytest.raises(RuntimeError):
        guide()


def test_unpack_latent():
    def model():
        return pyro.sample('x', dist.LKJCorrCholesky(2, torch.tensor(1.)))

    guide = AutoDiagonalNormal(model)
    assert guide()['x'].shape == model().shape
    latent = guide.sample_latent()
    assert list(guide._unpack_latent(latent))[0][1].shape == (1,)


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
])
def test_init_loc_fn(auto_class):

    def model():
        pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.MultivariateNormal(torch.zeros(5), torch.eye(5, 5)))

    inits = {"x": torch.randn(()), "y": torch.randn(5)}

    def init_loc_fn(site):
        return inits[site["name"]]

    guide = auto_class(model, init_loc_fn=init_loc_fn)
    guide()
    median = guide.median()
    assert_equal(median["x"], inits["x"])
    assert_equal(median["y"], inits["y"])


# testing helper
class AutoLowRankMultivariateNormal_100(AutoLowRankMultivariateNormal):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs, rank=100)


@pytest.mark.parametrize("init_scale", [1e-1, 1e-4, 1e-8])
@pytest.mark.parametrize("auto_class", [
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoLowRankMultivariateNormal_100,
])
def test_init_scale(auto_class, init_scale):

    def model():
        pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.MultivariateNormal(torch.zeros(5), torch.eye(5, 5)))
        with pyro.plate("plate", 100):
            pyro.sample("z", dist.Normal(0., 1.))

    guide = auto_class(model, init_scale=init_scale)
    guide()
    loc, scale = guide._loc_scale()
    scale_rms = scale.pow(2).mean().sqrt().item()
    assert init_scale * 0.5 < scale_rms < 2.0 * init_scale


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
    auto_guide_list_x,
    auto_guide_callable,
    auto_guide_module_callable,
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_mean),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_median),
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_median_module(auto_class, Elbo):

    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.x_loc = nn.Parameter(torch.tensor(1.))
            self.x_scale = PyroParam(torch.tensor(0.1), constraints.positive)

        def forward(self):
            pyro.sample("x", dist.Normal(self.x_loc, self.x_scale))
            pyro.sample("y", dist.Normal(2., 0.1))

    model = Model()
    guide = auto_class(model)
    infer = SVI(model, guide, Adam({'lr': 0.005}), Elbo(strict_enumeration_warning=False))
    for _ in range(20):
        infer.step()

    if auto_class is AutoLaplaceApproximation:
        guide = guide.laplace_approximation()

    median = guide.median()
    assert_equal(median["x"].detach(), torch.tensor(1.0), prec=0.1)
    assert_equal(median["y"].detach(), torch.tensor(2.0), prec=0.1)


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_autoguide(Elbo):

    class Model(PyroModule):
        def __init__(self):
            super().__init__()
            self.x_loc = nn.Parameter(torch.tensor(1.))
            self.x_scale = PyroParam(torch.tensor(0.1), constraints.positive)

        def forward(self):
            pyro.sample("x", dist.Normal(self.x_loc, self.x_scale))
            with pyro.plate("plate", 2):
                pyro.sample("y", dist.Normal(2., 0.1))

    model = Model()
    guide = nested_auto_guide_callable(model)

    # Check master ref for all nested components.
    for _, m in guide.named_modules():
        if m is guide:
            continue
        assert m.master is not None and m.master() is guide, "master ref wrong for {}".format(m._pyro_name)

    infer = SVI(model, guide, Adam({'lr': 0.005}), Elbo(strict_enumeration_warning=False))
    for _ in range(20):
        infer.step()

    guide_trace = poutine.trace(guide).get_trace()
    model_trace = poutine.trace(model).get_trace()
    check_model_guide_match(model_trace, guide_trace)
    assert all(p.startswith("AutoGuideList.0") or p.startswith("AutoGuideList.1.z")
               for p in guide_trace.param_nodes)
    stochastic_nodes = set(guide_trace.stochastic_nodes)
    assert "x" in stochastic_nodes
    assert "y" in stochastic_nodes
    # Only latent sampled is for the IAF.
    assert "_AutoGuideList.1.z_latent" in stochastic_nodes


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_mean),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_median),
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_linear_regression_smoke(auto_class, Elbo):
    N, D = 10, 3

    class RandomLinear(nn.Linear, PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
            self.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    class LinearRegression(PyroModule):
        def __init__(self):
            super().__init__()
            self.linear = RandomLinear(D, 1)

        def forward(self, x, y=None):
            mean = self.linear(x).squeeze(-1)
            sigma = pyro.sample("sigma", dist.LogNormal(0., 1.))
            with pyro.plate('plate', N):
                return pyro.sample('obs', dist.Normal(mean, sigma), obs=y)

    x, y = torch.randn(N, D), torch.randn(N)
    model = LinearRegression()
    guide = auto_class(model)
    infer = SVI(model, guide, Adam({'lr': 0.005}), Elbo(strict_enumeration_warning=False))
    infer.step(x, y)


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_mean),
    functools.partial(AutoDiagonalNormal, init_loc_fn=init_to_median),
])
def test_predictive(auto_class):
    N, D = 3, 2

    class RandomLinear(nn.Linear, PyroModule):
        def __init__(self, in_features, out_features):
            super().__init__(in_features, out_features)
            self.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
            self.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    class LinearRegression(PyroModule):
        def __init__(self):
            super().__init__()
            self.linear = RandomLinear(D, 1)

        def forward(self, x, y=None):
            mean = self.linear(x).squeeze(-1)
            sigma = pyro.sample("sigma", dist.LogNormal(0., 1.))
            with pyro.plate('plate', N):
                return pyro.sample('obs', dist.Normal(mean, sigma), obs=y)

    x, y = torch.randn(N, D), torch.randn(N)
    model = LinearRegression()
    guide = auto_class(model)
    # XXX: Record `y` as observed in the prototype trace
    # Is there a better pattern to follow?
    guide(x, y=y)
    # Test predictive module
    model_trace = poutine.trace(model).get_trace(x, y=None)
    predictive = Predictive(model, guide=guide, num_samples=10)
    pyro.set_rng_seed(0)
    samples = predictive(x)
    for site in prune_subsample_sites(model_trace).stochastic_nodes:
        assert site in samples
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        traced_predictive = torch.jit.trace_module(predictive, {"call": (x,)})
    f = io.BytesIO()
    torch.jit.save(traced_predictive, f)
    f.seek(0)
    predictive_deser = torch.jit.load(f)
    pyro.set_rng_seed(0)
    samples_deser = predictive_deser.call(x)
    # Note that the site values are different in the serialized guide
    assert len(samples) == len(samples_deser)


@pytest.mark.parametrize("init_fn", [None, init_to_mean, init_to_median])
@pytest.mark.parametrize("auto_class", [AutoDelta, AutoNormal, AutoGuideList])
def test_subsample_guide(auto_class, init_fn):

    # The model from tutorial/source/easyguide.ipynb
    def model(batch, subsample, full_size):
        num_time_steps = len(batch)
        result = [None] * num_time_steps
        drift = pyro.sample("drift", dist.LogNormal(-1, 0.5))
        plate = pyro.plate("data", full_size, subsample=subsample)
        assert plate.size == 50
        with plate:
            z = 0.
            for t in range(num_time_steps):
                z = pyro.sample("state_{}".format(t), dist.Normal(z, drift))
                result[t] = pyro.sample("obs_{}".format(t), dist.Bernoulli(logits=z),
                                        obs=batch[t])

        return torch.stack(result)

    def create_plates(batch, subsample, full_size):
        return pyro.plate("data", full_size, subsample=subsample)

    if auto_class == AutoGuideList:
        guide = AutoGuideList(model, create_plates=create_plates)
        guide.add(AutoDelta(poutine.block(model, expose=["drift"])))
        guide.add(AutoNormal(poutine.block(model, hide=["drift"])))
    else:
        guide = auto_class(model, create_plates=create_plates)

    full_size = 50
    batch_size = 20
    num_time_steps = 8
    pyro.set_rng_seed(123456789)
    data = model([None] * num_time_steps, torch.arange(full_size), full_size)
    assert data.shape == (num_time_steps, full_size)

    pyro.get_param_store().clear()
    pyro.set_rng_seed(123456789)
    svi = SVI(model, guide, Adam({"lr": 0.02}), Trace_ELBO())
    for epoch in range(2):
        beg = 0
        while beg < full_size:
            end = min(full_size, beg + batch_size)
            subsample = torch.arange(beg, end)
            batch = data[:, beg:end]
            beg = end
            svi.step(batch, subsample, full_size=full_size)


@pytest.mark.parametrize("independent", [True, False], ids=["independent", "dependent"])
@pytest.mark.parametrize("auto_class", [AutoDelta, AutoNormal])
def test_subsample_guide_2(auto_class, independent):

    # Simplified from Model2 in tutorial/source/forecasting_iii.ipynb
    def model(data):
        size, size = data.shape
        origin_plate = pyro.plate("origin", size, dim=-2)
        destin_plate = pyro.plate("destin", size, dim=-1)
        with origin_plate, destin_plate:
            batch = pyro.subsample(data, event_dim=0)
            assert batch.size(0) == batch.size(1), batch.shape
            pyro.sample("obs", dist.Normal(0, 1), obs=batch)

    def create_plates(data):
        size, size = data.shape
        origin_plate = pyro.plate("origin", size, subsample_size=5, dim=-2)
        if independent:
            destin_plate = pyro.plate("destin", size, subsample_size=5, dim=-1)
        else:
            with origin_plate as subsample:
                pass
            destin_plate = pyro.plate("destin", size, subsample=subsample, dim=-1)
        return origin_plate, destin_plate

    guide = auto_class(model, create_plates=create_plates)
    svi = SVI(model, guide, Adam({"lr": 0.01}), Trace_ELBO())

    data = torch.randn(10, 10)
    for step in range(2):
        svi.step(data)
