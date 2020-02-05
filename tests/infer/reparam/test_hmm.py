# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import LinearHMMReparam, StableReparam, StudentTReparam, SymmetricStableReparam
from tests.ops.gaussian import random_mvn


def random_studentt(shape):
    df = torch.rand(shape).exp()
    loc = torch.randn(shape)
    scale = torch.rand(shape).exp()
    return dist.StudentT(df, loc, scale)


def random_stable(shape, stability, skew=None):
    if skew is None:
        skew = dist.Uniform(-1, 1).sample(shape)
    scale = torch.rand(shape).exp()
    loc = torch.randn(shape)
    return dist.Stable(stability, skew, scale, loc)


@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("hidden_dim", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
def test_transformed_hmm_shape(batch_shape, duration, hidden_dim, obs_dim):
    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_mat = torch.randn(batch_shape + (duration, hidden_dim, hidden_dim))
    trans_dist = random_mvn(batch_shape + (duration,), hidden_dim)
    obs_mat = torch.randn(batch_shape + (duration, hidden_dim, obs_dim))
    obs_dist = dist.LogNormal(torch.randn(batch_shape + (duration, obs_dim)),
                              torch.rand(batch_shape + (duration, obs_dim)).exp()).to_event(1)
    hmm = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=duration)

    def model(data=None):
        with pyro.plate_stack("plates", batch_shape):
            return pyro.sample("x", hmm, obs=data)

    data = model()
    with poutine.trace() as tr:
        with poutine.reparam(config={"x": LinearHMMReparam()}):
            model(data)
    fn = tr.trace.nodes["x"]["fn"]
    assert isinstance(fn, dist.TransformedDistribution)
    assert isinstance(fn.base_dist, dist.GaussianHMM)
    tr.trace.compute_log_prob()  # smoke test only


@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("hidden_dim", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
def test_studentt_hmm_shape(batch_shape, duration, hidden_dim, obs_dim):
    init_dist = random_studentt(batch_shape + (hidden_dim,)).to_event(1)
    trans_mat = torch.randn(batch_shape + (duration, hidden_dim, hidden_dim))
    trans_dist = random_studentt(batch_shape + (duration, hidden_dim)).to_event(1)
    obs_mat = torch.randn(batch_shape + (duration, hidden_dim, obs_dim))
    obs_dist = random_studentt(batch_shape + (duration, obs_dim)).to_event(1)
    hmm = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=duration)

    def model(data=None):
        with pyro.plate_stack("plates", batch_shape):
            return pyro.sample("x", hmm, obs=data)

    data = model()
    rep = StudentTReparam()
    with poutine.trace() as tr:
        with poutine.reparam(config={"x": LinearHMMReparam(rep, rep, rep)}):
            model(data)
    assert isinstance(tr.trace.nodes["x"]["fn"], dist.GaussianHMM)
    tr.trace.compute_log_prob()  # smoke test only


@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("hidden_dim", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize("skew", [0, None], ids=["symmetric", "skewed"])
def test_stable_hmm_shape(skew, batch_shape, duration, hidden_dim, obs_dim):
    stability = dist.Uniform(0.5, 2).sample(batch_shape)
    init_dist = random_stable(batch_shape + (hidden_dim,),
                              stability.unsqueeze(-1), skew=skew).to_event(1)
    trans_mat = torch.randn(batch_shape + (duration, hidden_dim, hidden_dim))
    trans_dist = random_stable(batch_shape + (duration, hidden_dim),
                               stability.unsqueeze(-1).unsqueeze(-1), skew=skew).to_event(1)
    obs_mat = torch.randn(batch_shape + (duration, hidden_dim, obs_dim))
    obs_dist = random_stable(batch_shape + (duration, obs_dim),
                             stability.unsqueeze(-1).unsqueeze(-1), skew=skew).to_event(1)
    hmm = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=duration)
    assert hmm.batch_shape == batch_shape
    assert hmm.event_shape == (duration, obs_dim)

    def model(data=None):
        with pyro.plate_stack("plates", batch_shape):
            return pyro.sample("x", hmm, obs=data)

    data = torch.randn(duration, obs_dim)
    rep = SymmetricStableReparam() if skew == 0 else StableReparam()
    with poutine.trace() as tr:
        with poutine.reparam(config={"x": LinearHMMReparam(rep, rep, rep)}):
            model(data)
    assert isinstance(tr.trace.nodes["x"]["fn"], dist.GaussianHMM)
    tr.trace.compute_log_prob()  # smoke test only


@pytest.mark.parametrize("duration", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("hidden_dim", [1, 3])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3)], ids=str)
def test_stable_hmm_shape_error(batch_shape, duration, hidden_dim, obs_dim):
    stability = dist.Uniform(0.5, 2).sample(batch_shape)
    init_dist = random_stable(batch_shape + (hidden_dim,),
                              stability.unsqueeze(-1)).to_event(1)
    trans_mat = torch.randn(batch_shape + (1, hidden_dim, hidden_dim))
    trans_dist = random_stable(batch_shape + (1, hidden_dim),
                               stability.unsqueeze(-1).unsqueeze(-1)).to_event(1)
    obs_mat = torch.randn(batch_shape + (1, hidden_dim, obs_dim))
    obs_dist = random_stable(batch_shape + (1, obs_dim),
                             stability.unsqueeze(-1).unsqueeze(-1)).to_event(1)
    hmm = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)
    assert hmm.batch_shape == batch_shape
    assert hmm.event_shape == (1, obs_dim)

    def model(data=None):
        with pyro.plate_stack("plates", batch_shape):
            return pyro.sample("x", hmm, obs=data)

    data = torch.randn(duration, obs_dim)
    rep = StableReparam()
    with poutine.reparam(config={"x": LinearHMMReparam(rep, rep, rep)}):
        with pytest.raises(ValueError):
            model(data)
