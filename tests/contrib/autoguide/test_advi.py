from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.autoguide import (AutoCallable, AutoDelta, AutoDiagonalNormal, AutoDiscreteParallel, AutoGuideList,
                                    AutoLowRankMultivariateNormal, AutoMultivariateNormal, AutoIAFNormal)
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from tests.common import assert_equal


@pytest.mark.parametrize("auto_class", [
    AutoDiagonalNormal,
    AutoMultivariateNormal,
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

    assert '_auto_latent' not in model_trace.nodes
    assert model_trace.nodes['z']['log_prob_sum'].item() != 0.0
    assert guide_trace.nodes['_auto_latent']['log_prob_sum'].item() != 0.0
    assert guide_trace.nodes['z']['log_prob_sum'].item() == 0.0


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
])
def test_shapes(auto_class, Elbo):

    def model():
        pyro.sample("z1", dist.Normal(0.0, 1.0))
        pyro.sample("z2", dist.Normal(torch.zeros(2), torch.ones(2)).independent(1))
        with pyro.iarange("iarange", 3):
            pyro.sample("z3", dist.Normal(torch.zeros(3), torch.ones(3)))

    guide = auto_class(model)
    elbo = Elbo(strict_enumeration_warning=False)
    loss = elbo.loss(model, guide)
    assert np.isfinite(loss), loss


@pytest.mark.xfail(reason="irange is not yet supported")
@pytest.mark.parametrize('auto_class', [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO])
def test_irange_smoke(auto_class, Elbo):

    def model():
        x = pyro.sample("x", dist.Normal(0, 1))
        assert x.shape == ()

        for i in pyro.irange("irange", 3):
            y = pyro.sample("y_{}".format(i), dist.Normal(0, 1).expand_by([2, 1 + i, 2]).independent(3))
            assert y.shape == (2, 1 + i, 2)

        z = pyro.sample("z", dist.Normal(0, 1).expand_by([2]).independent(1))
        assert z.shape == (2,)

        pyro.sample("obs", dist.Bernoulli(0.1), obs=torch.tensor(0))

    guide = auto_class(model)
    infer = SVI(model, guide, Adam({"lr": 1e-6}), Elbo(strict_enumeration_warning=False))
    infer.step()


def auto_guide_list_x(model):
    guide = AutoGuideList(model)
    guide.add(AutoDelta(poutine.block(model, expose=["x"])))
    guide.add(AutoDiagonalNormal(poutine.block(model, hide=["x"])))
    return guide


def auto_guide_callable(model):
    def guide_x():
        x_loc = pyro.param("x_loc", torch.tensor(1.))
        x_scale = pyro.param("x_scale", torch.tensor(2.), constraint=constraints.positive)
        pyro.sample("x", dist.Normal(x_loc, x_scale))

    def median_x():
        return {"x": pyro.param("x_loc", torch.tensor(1.))}

    guide = AutoGuideList(model)
    guide.add(AutoCallable(model, guide_x, median_x))
    guide.add(AutoDiagonalNormal(poutine.block(model, hide=["x"])))
    return guide


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    auto_guide_list_x,
    auto_guide_callable,
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_median(auto_class, Elbo):

    def model():
        pyro.sample("x", dist.Normal(0.0, 1.0))
        pyro.sample("y", dist.LogNormal(0.0, 1.0))
        pyro.sample("z", dist.Beta(2.0, 2.0))

    guide = auto_class(model)
    infer = SVI(model, guide, Adam({'lr': 0.005}), Elbo(strict_enumeration_warning=False))
    for _ in range(800):
        infer.step()

    median = guide.median()
    assert_equal(median["x"], torch.tensor(0.0), prec=0.1)
    if auto_class is AutoDelta:
        assert_equal(median["y"], torch.tensor(-1.0).exp(), prec=0.1)
    else:
        assert_equal(median["y"], torch.tensor(1.0), prec=0.1)
    assert_equal(median["z"], torch.tensor(0.5), prec=0.1)


@pytest.mark.parametrize("auto_class", [
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_quantiles(auto_class, Elbo):

    def model():
        pyro.sample("x", dist.Normal(0.0, 1.0))
        pyro.sample("y", dist.LogNormal(0.0, 1.0))
        pyro.sample("z", dist.Beta(2.0, 2.0))

    guide = auto_class(model)
    infer = SVI(model, guide, Adam({'lr': 0.01}), Elbo(strict_enumeration_warning=False))
    for _ in range(100):
        infer.step()

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
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
])
def test_discrete_parallel(continuous_class):
    K = 2
    data = torch.tensor([0., 1., 10., 11., 12.])

    def model(data):
        weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
        locs = pyro.sample('locs', dist.Normal(0, 10).expand_by([K]).independent(1))
        scale = pyro.sample('scale', dist.LogNormal(0, 1))

        with pyro.iarange('data', len(data)):
            weights = weights.expand(torch.Size((len(data),)) + weights.shape)
            assignment = pyro.sample('assignment', dist.Categorical(weights))
            pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

    guide = AutoGuideList(model)
    guide.add(continuous_class(poutine.block(model, hide=["assignment"])))
    guide.add(AutoDiscreteParallel(poutine.block(model, expose=["assignment"])))

    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    loss = elbo.loss_and_grads(model, guide, data)
    assert np.isfinite(loss), loss


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
])
def test_guide_list(auto_class):

    def model():
        pyro.sample("x", dist.Normal(0., 1.).expand([2]))
        pyro.sample("y", dist.MultivariateNormal(torch.zeros(5), torch.eye(5, 5)))

    guide = AutoGuideList(model)
    guide.add(auto_class(poutine.block(model, expose=["x"]), prefix="auto_x"))
    guide.add(auto_class(poutine.block(model, expose=["y"]), prefix="auto_y"))
    guide()


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
])
def test_callable(auto_class):

    def model():
        pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.MultivariateNormal(torch.zeros(5), torch.eye(5, 5)))

    def guide_x():
        x_loc = pyro.param("x_loc", torch.tensor(0.))
        pyro.sample("x", dist.Delta(x_loc))

    guide = AutoGuideList(model)
    guide.add(guide_x)
    guide.add(auto_class(poutine.block(model, expose=["y"]), prefix="auto_y"))
    values = guide()
    assert set(values) == set(["y"])


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
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
    guide.add(guide_x)
    guide.add(auto_class(poutine.block(model, expose=["y"]), prefix="auto_y"))
    values = guide()
    assert set(values) == set(["x", "y"])


def test_empty_model_error():
    def model():
        pass
    guide = AutoDiagonalNormal(model)
    with pytest.raises(RuntimeError):
        guide()
