# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import LinearHMMReparam, StableReparam, StudentTReparam, SymmetricStableReparam
from tests.common import assert_close
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
    assert tr.trace.nodes["x_init_gamma"]["fn"].event_shape == (hidden_dim,)
    assert tr.trace.nodes["x_trans_gamma"]["fn"].event_shape == (duration, hidden_dim)
    assert tr.trace.nodes["x_obs_gamma"]["fn"].event_shape == (duration, obs_dim)
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
@pytest.mark.parametrize("skew", [0, None], ids=["symmetric", "skewed"])
def test_independent_hmm_shape(skew, batch_shape, duration, hidden_dim, obs_dim):
    base_batch_shape = batch_shape + (obs_dim,)
    stability = dist.Uniform(0.5, 2).sample(base_batch_shape)
    init_dist = random_stable(base_batch_shape + (hidden_dim,),
                              stability.unsqueeze(-1), skew=skew).to_event(1)
    trans_mat = torch.randn(base_batch_shape + (duration, hidden_dim, hidden_dim))
    trans_dist = random_stable(base_batch_shape + (duration, hidden_dim),
                               stability.unsqueeze(-1).unsqueeze(-1), skew=skew).to_event(1)
    obs_mat = torch.randn(base_batch_shape + (duration, hidden_dim, 1))
    obs_dist = random_stable(base_batch_shape + (duration, 1),
                             stability.unsqueeze(-1).unsqueeze(-1), skew=skew).to_event(1)
    hmm = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=duration)
    assert hmm.batch_shape == base_batch_shape
    assert hmm.event_shape == (duration, 1)

    hmm = dist.IndependentHMM(hmm)
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
    assert isinstance(tr.trace.nodes["x"]["fn"], dist.IndependentHMM)
    tr.trace.compute_log_prob()  # smoke test only


# Test helper to extract a few fractional moments from joint samples.
# This uses fractional moments because Stable variance is infinite.
def get_hmm_moments(samples):
    loc = samples.median(0).values
    delta = samples - loc
    cov = (delta.unsqueeze(-1) * delta.unsqueeze(-2)).sqrt().mean(0)
    scale = cov.diagonal(dim1=-2, dim2=-1)
    sigma = scale.sqrt()
    corr = cov / (sigma.unsqueeze(-1) * sigma.unsqueeze(-2))
    return loc, scale, corr


@pytest.mark.parametrize("duration", [1, 2, 3])
@pytest.mark.parametrize("obs_dim", [1, 2])
@pytest.mark.parametrize("hidden_dim", [1, 2])
@pytest.mark.parametrize("stability", [1.9, 1.6])
@pytest.mark.parametrize("skew", [0, None], ids=["symmetric", "skewed"])
def test_stable_hmm_distribution(stability, skew, duration, hidden_dim, obs_dim):
    init_dist = random_stable((hidden_dim,), stability, skew=skew).to_event(1)
    trans_mat = torch.randn(duration, hidden_dim, hidden_dim)
    trans_dist = random_stable((duration, hidden_dim), stability, skew=skew).to_event(1)
    obs_mat = torch.randn(duration, hidden_dim, obs_dim)
    obs_dist = random_stable((duration, obs_dim), stability, skew=skew).to_event(1)
    hmm = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=duration)

    num_samples = 200000
    expected_samples = hmm.sample([num_samples]).reshape(num_samples, duration * obs_dim)
    expected_loc, expected_scale, expected_corr = get_hmm_moments(expected_samples)

    rep = SymmetricStableReparam() if skew == 0 else StableReparam()
    with pyro.plate("samples", num_samples):
        with poutine.reparam(config={"x": LinearHMMReparam(rep, rep, rep)}):
            actual_samples = pyro.sample("x", hmm).reshape(num_samples, duration * obs_dim)
    actual_loc, actual_scale, actual_corr = get_hmm_moments(actual_samples)

    assert_close(actual_loc, expected_loc, atol=0.05, rtol=0.05)
    assert_close(actual_scale, expected_scale, atol=0.05, rtol=0.05)
    assert_close(actual_corr, expected_corr, atol=0.01)


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
