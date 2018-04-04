from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import ELBO, SVI, ADVIDiagonalNormal, ADVIMultivariateNormal
from pyro.optim import Adam
from tests.common import assert_equal


@pytest.mark.parametrize("advi_class", [ADVIMultivariateNormal, ADVIDiagonalNormal])
def test_scores(advi_class):
    def model():
        pyro.sample("z", dist.Normal(0.0, 1.0))

    advi = advi_class(model)
    guide_trace = poutine.trace(advi.guide).get_trace()
    model_trace = poutine.trace(poutine.replay(advi.model, guide_trace)).get_trace()

    guide_trace.compute_log_prob()
    model_trace.compute_log_prob()

    assert model_trace.nodes['_advi_latent']['log_prob_sum'].item() == 0.0
    assert model_trace.nodes['z']['log_prob_sum'].item() != 0.0
    assert guide_trace.nodes['_advi_latent']['log_prob_sum'].item() != 0.0
    assert guide_trace.nodes['z']['log_prob_sum'].item() == 0.0


@pytest.mark.parametrize("trace_graph,enum_discrete",
                         [(False, False), (True, False), (False, True)],
                         ids=["Trace", "TraceGraph", "TraceEnum"])
@pytest.mark.parametrize("advi_class", [ADVIMultivariateNormal, ADVIDiagonalNormal])
def test_shapes(advi_class, trace_graph, enum_discrete):

    def model():
        pyro.sample("z1", dist.Normal(0.0, 1.0))
        pyro.sample("z2", dist.Normal(torch.zeros(2), torch.ones(2)).reshape(extra_event_dims=1))
        with pyro.iarange("iarange", 3):
            pyro.sample("z3", dist.Normal(torch.zeros(3), torch.ones(3)))

    advi = advi_class(model)
    elbo = ELBO.make(trace_graph=trace_graph, enum_discrete=enum_discrete)
    loss = elbo.loss(advi.model, advi.guide)
    assert np.isfinite(loss), loss


@pytest.mark.parametrize("advi_class", [ADVIMultivariateNormal, ADVIDiagonalNormal])
def test_median(advi_class):

    def model():
        pyro.sample("x", dist.Normal(0.0, 1.0))
        pyro.sample("y", dist.LogNormal(0.0, 1.0))
        pyro.sample("z", dist.Beta(2.0, 2.0))

    advi = advi_class(model)
    infer = SVI(advi.model, advi.guide, Adam({'lr': 0.01}), 'ELBO')
    for _ in range(100):
        infer.step()

    median = advi.median()
    assert_equal(median["x"], torch.tensor(0.0), prec=0.1)
    assert_equal(median["y"], torch.tensor(1.0), prec=0.1)
    assert_equal(median["z"], torch.tensor(0.5), prec=0.1)


@pytest.mark.parametrize("advi_class", [ADVIMultivariateNormal, ADVIDiagonalNormal])
def test_quantiles(advi_class):

    def model():
        pyro.sample("x", dist.Normal(0.0, 1.0))
        pyro.sample("y", dist.LogNormal(0.0, 1.0))
        pyro.sample("z", dist.Beta(2.0, 2.0))

    advi = advi_class(model)
    infer = SVI(advi.model, advi.guide, Adam({'lr': 0.01}), 'ELBO')
    for _ in range(100):
        infer.step()

    quantiles = advi.quantiles([0.1, 0.5, 0.9])
    median = advi.median()
    for name in ["x", "y", "z"]:
        assert_equal(median[name], quantiles[name][1])

    assert torch.tensor(-3.0) < quantiles["x"][0]
    assert quantiles["x"][0] + 1.0 < quantiles["x"][1]
    assert quantiles["x"][1] + 1.0 < quantiles["x"][2]
    assert quantiles["x"][2] < 3.0

    assert torch.tensor(0.01) < quantiles["y"][0]
    assert quantiles["y"][0] * 2.0 < quantiles["y"][1]
    assert quantiles["y"][1] * 2.0 < quantiles["y"][2]
    assert quantiles["y"][2] < 100.0

    assert torch.tensor(0.01) < quantiles["z"][0]
    assert quantiles["z"][0] + 0.1 < quantiles["z"][1]
    assert quantiles["z"][1] + 0.1 < quantiles["z"][2]
    assert quantiles["z"][2] < 0.99
