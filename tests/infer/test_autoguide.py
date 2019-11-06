import functools

import numpy as np
import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO
from pyro.infer.autoguide import (AutoCallable, AutoDelta, AutoDiagonalNormal, AutoDiscreteParallel, AutoGuideList,
                                  AutoIAFNormal, AutoLaplaceApproximation, AutoLowRankMultivariateNormal,
                                  AutoMultivariateNormal, init_to_feasible, init_to_mean, init_to_median,
                                  init_to_sample)
from pyro.optim import Adam
from tests.common import assert_close, assert_equal


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
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
])
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
    guide.add(AutoDelta(poutine.block(model, expose=["x"])))
    guide.add(AutoDiagonalNormal(poutine.block(model, hide=["x"])))
    return guide


def auto_guide_callable(model):
    def guide_x():
        x_loc = pyro.param("x_loc", torch.tensor(1.))
        x_scale = pyro.param("x_scale", torch.tensor(.1), constraint=constraints.positive)
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
    AutoLaplaceApproximation,
    auto_guide_list_x,
    auto_guide_callable,
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
    infer = SVI(model, guide, Adam({'lr': 0.005}), Elbo(strict_enumeration_warning=False))
    for _ in range(800):
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
    AutoDiagonalNormal,
    AutoMultivariateNormal,
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
    infer = SVI(model, guide, Adam({'lr': 0.01}), Elbo(strict_enumeration_warning=False))
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
    guide.add(continuous_class(poutine.block(model, hide=["assignment"])))
    guide.add(AutoDiscreteParallel(poutine.block(model, expose=["assignment"])))

    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    loss = elbo.loss_and_grads(model, guide, data)
    assert np.isfinite(loss), loss


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoIAFNormal,
    AutoLaplaceApproximation,
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
    guide.add(guide_x)
    guide.add(auto_class(poutine.block(model, expose=["y"]), prefix="auto_y"))
    values = guide()
    assert set(values) == set(["y"])


@pytest.mark.parametrize("auto_class", [
    AutoDelta,
    AutoDiagonalNormal,
    AutoMultivariateNormal,
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
