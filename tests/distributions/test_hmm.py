# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import operator
from functools import reduce

import opt_einsum
import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.hmm import (_sequential_gamma_gaussian_tensordot, _sequential_gaussian_filter_sample,
                                    _sequential_gaussian_tensordot, _sequential_logmatmulexp)
from pyro.distributions.util import broadcast_shape
from pyro.infer import TraceEnum_ELBO, config_enumerate
from pyro.ops.gamma_gaussian import (gamma_and_mvn_to_gamma_gaussian, gamma_gaussian_tensordot,
                                     matrix_and_mvn_to_gamma_gaussian)
from pyro.ops.gaussian import gaussian_tensordot, matrix_and_mvn_to_gaussian, mvn_to_gaussian
from pyro.ops.indexing import Vindex
from tests.common import assert_close
from tests.ops.gamma_gaussian import assert_close_gamma_gaussian, random_gamma, random_gamma_gaussian
from tests.ops.gaussian import assert_close_gaussian, random_gaussian, random_mvn

logger = logging.getLogger(__name__)


def check_expand(old_dist, old_data):
    new_batch_shape = (2,) + old_dist.batch_shape
    new_dist = old_dist.expand(new_batch_shape)
    assert new_dist.batch_shape == new_batch_shape

    old_log_prob = new_dist.log_prob(old_data)
    assert old_log_prob.shape == new_batch_shape

    new_data = old_data.expand(new_batch_shape + new_dist.event_shape)
    new_log_prob = new_dist.log_prob(new_data)
    assert_close(old_log_prob, new_log_prob)
    assert new_dist.log_prob(new_data).shape == new_batch_shape


@pytest.mark.parametrize('num_steps', list(range(1, 20)))
@pytest.mark.parametrize('state_dim', [2, 3])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 4)], ids=str)
def test_sequential_logmatmulexp(batch_shape, state_dim, num_steps):
    logits = torch.randn(batch_shape + (num_steps, state_dim, state_dim))
    actual = _sequential_logmatmulexp(logits)
    assert actual.shape == batch_shape + (state_dim, state_dim)

    # Check against einsum.
    operands = list(logits.unbind(-3))
    symbol = (opt_einsum.get_symbol(i) for i in range(1000))
    batch_symbols = ''.join(next(symbol) for _ in batch_shape)
    state_symbols = [next(symbol) for _ in range(num_steps + 1)]
    equation = (','.join(batch_symbols + state_symbols[t] + state_symbols[t + 1]
                         for t in range(num_steps)) +
                '->' + batch_symbols + state_symbols[0] + state_symbols[-1])
    expected = opt_einsum.contract(equation, *operands, backend='pyro.ops.einsum.torch_log')
    assert_close(actual, expected)


@pytest.mark.parametrize('num_steps', list(range(1, 20)))
@pytest.mark.parametrize('state_dim', [1, 2, 3])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 4)], ids=str)
def test_sequential_gaussian_tensordot(batch_shape, state_dim, num_steps):
    g = random_gaussian(batch_shape + (num_steps,), state_dim + state_dim)
    actual = _sequential_gaussian_tensordot(g)
    assert actual.dim() == g.dim()
    assert actual.batch_shape == batch_shape

    # Check against hand computation.
    expected = g[..., 0]
    for t in range(1, num_steps):
        expected = gaussian_tensordot(expected, g[..., t], state_dim)
    assert_close_gaussian(actual, expected)


@pytest.mark.parametrize('num_steps', list(range(1, 20)))
@pytest.mark.parametrize('state_dim', [1, 2, 3])
@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('sample_shape', [(), (4,), (3, 2)], ids=str)
def test_sequential_gaussian_filter_sample(sample_shape, batch_shape, state_dim, num_steps):
    init = random_gaussian(batch_shape, state_dim)
    trans = random_gaussian(batch_shape + (num_steps,), state_dim + state_dim)
    sample = _sequential_gaussian_filter_sample(init, trans, sample_shape)
    assert sample.shape == sample_shape + batch_shape + (num_steps, state_dim)


@pytest.mark.parametrize('num_steps', list(range(1, 20)))
@pytest.mark.parametrize('state_dim', [1, 2, 3])
@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 4)], ids=str)
def test_sequential_gamma_gaussian_tensordot(batch_shape, state_dim, num_steps):
    g = random_gamma_gaussian(batch_shape + (num_steps,), state_dim + state_dim)
    actual = _sequential_gamma_gaussian_tensordot(g)
    assert actual.dim() == g.dim()
    assert actual.batch_shape == batch_shape

    # Check against hand computation.
    expected = g[..., 0]
    for t in range(1, num_steps):
        expected = gamma_gaussian_tensordot(expected, g[..., t], state_dim)
    assert_close_gamma_gaussian(actual, expected)


@pytest.mark.parametrize('state_dim', [2, 3])
@pytest.mark.parametrize('event_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('ok,init_shape,trans_shape,obs_shape', [
    (True, (), (), (1,)),
    (True, (), (1,), (1,)),
    (True, (), (), (7,)),
    (True, (), (7,), (7,)),
    (True, (), (1,), (7,)),
    (True, (), (7,), (11, 7)),
    (True, (), (11, 7), (7,)),
    (True, (), (11, 7), (11, 7)),
    (True, (11,), (7,), (7,)),
    (True, (11,), (7,), (11, 7)),
    (True, (11,), (11, 7), (7,)),
    (True, (11,), (11, 7), (11, 7)),
    (True, (4, 1, 1), (3, 1, 7), (2, 7)),
    (False, (), (1,), ()),
    (False, (), (7,), ()),
    (False, (), (7,), (1,)),
    (False, (), (7,), (6,)),
    (False, (3,), (4, 7), (7,)),
    (False, (3,), (7,), (4, 7)),
    (False, (), (3, 7), (4, 7)),
], ids=str)
def test_discrete_hmm_shape(ok, init_shape, trans_shape, obs_shape, event_shape, state_dim):
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    obs_logits = torch.randn(obs_shape + (state_dim,) + event_shape)
    obs_dist = dist.Bernoulli(logits=obs_logits).to_event(len(event_shape))
    data = obs_dist.sample()[(slice(None),) * len(obs_shape) + (0,)]

    if not ok:
        with pytest.raises(ValueError):
            d = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
            d.log_prob(data)
        return

    d = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)

    actual = d.log_prob(data)
    expected_shape = broadcast_shape(init_shape, trans_shape[:-1], obs_shape[:-1])
    assert actual.shape == expected_shape
    check_expand(d, data)

    final = d.filter(data)
    assert isinstance(final, dist.Categorical)
    assert final.batch_shape == d.batch_shape
    assert final.event_shape == ()
    assert final.support.upper_bound == state_dim - 1


@pytest.mark.parametrize('event_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('state_dim', [2, 3])
@pytest.mark.parametrize('num_steps', [1, 2, 3])
@pytest.mark.parametrize('init_shape,trans_shape,obs_shape', [
    ((), (), ()),
    ((), (1,), ()),
    ((), (), (1,)),
    ((), (1,), (7, 1)),
    ((), (7, 1), (1,)),
    ((), (7, 1), (7, 1)),
    ((7,), (1,), (1,)),
    ((7,), (1,), (7, 1)),
    ((7,), (7, 1), (1,)),
    ((7,), (7, 1), (7, 1)),
    ((4, 1, 1), (3, 1, 1), (2, 1)),
], ids=str)
def test_discrete_hmm_homogeneous_trick(init_shape, trans_shape, obs_shape, event_shape, state_dim, num_steps):
    batch_shape = broadcast_shape(init_shape, trans_shape[:-1], obs_shape[:-1])
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    obs_logits = torch.randn(obs_shape + (state_dim,) + event_shape)
    obs_dist = dist.Bernoulli(logits=obs_logits).to_event(len(event_shape))

    d = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    assert d.event_shape == (1,) + event_shape

    data = obs_dist.expand(batch_shape + (num_steps, state_dim)).sample()
    data = data[(slice(None),) * (len(batch_shape) + 1) + (0,)]
    assert data.shape == batch_shape + (num_steps,) + event_shape
    actual = d.log_prob(data)
    assert actual.shape == batch_shape


def empty_guide(*args, **kwargs):
    pass


@pytest.mark.parametrize('num_steps', list(range(1, 10)))
def test_discrete_hmm_categorical(num_steps):
    state_dim = 3
    obs_dim = 4
    init_logits = torch.randn(state_dim)
    trans_logits = torch.randn(num_steps, state_dim, state_dim)
    obs_dist = dist.Categorical(logits=torch.randn(num_steps, state_dim, obs_dim))
    d = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    data = dist.Categorical(logits=torch.zeros(num_steps, obs_dim)).sample()
    actual = d.log_prob(data)
    assert actual.shape == d.batch_shape
    check_expand(d, data)

    # Check loss against TraceEnum_ELBO.
    @config_enumerate
    def model(data):
        x = pyro.sample("x_init", dist.Categorical(logits=init_logits))
        for t in range(num_steps):
            x = pyro.sample("x_{}".format(t),
                            dist.Categorical(logits=Vindex(trans_logits)[..., t, x, :]))
            pyro.sample("obs_{}".format(t),
                        dist.Categorical(logits=Vindex(obs_dist.logits)[..., t, x, :]),
                        obs=data[..., t])

    expected_loss = TraceEnum_ELBO().loss(model, empty_guide, data)
    actual_loss = -float(actual.sum())
    assert_close(actual_loss, expected_loss)


@pytest.mark.parametrize('num_steps', list(range(1, 10)))
def test_discrete_hmm_diag_normal(num_steps):
    state_dim = 3
    event_size = 2
    init_logits = torch.randn(state_dim)
    trans_logits = torch.randn(num_steps, state_dim, state_dim)
    loc = torch.randn(num_steps, state_dim, event_size)
    scale = torch.randn(num_steps, state_dim, event_size).exp()
    obs_dist = dist.Normal(loc, scale).to_event(1)
    d = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    data = obs_dist.sample()[:, 0]
    actual = d.log_prob(data)
    assert actual.shape == d.batch_shape
    check_expand(d, data)

    # Check loss against TraceEnum_ELBO.
    @config_enumerate
    def model(data):
        x = pyro.sample("x_init", dist.Categorical(logits=init_logits))
        for t in range(num_steps):
            x = pyro.sample("x_{}".format(t),
                            dist.Categorical(logits=Vindex(trans_logits)[..., t, x, :]))
            pyro.sample("obs_{}".format(t),
                        dist.Normal(Vindex(loc)[..., t, x, :],
                                    Vindex(scale)[..., t, x, :]).to_event(1),
                        obs=data[..., t, :])

    expected_loss = TraceEnum_ELBO().loss(model, empty_guide, data)
    actual_loss = -float(actual.sum())
    assert_close(actual_loss, expected_loss)


@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 3])
@pytest.mark.parametrize('init_shape,trans_mat_shape,trans_mvn_shape,obs_mat_shape,obs_mvn_shape', [
    ((), (), (), (), ()),
    ((), (6,), (), (), ()),
    ((), (), (6,), (), ()),
    ((), (), (), (6,), ()),
    ((), (), (), (), (6,)),
    ((), (6,), (6,), (6,), (6,)),
    ((5,), (6,), (), (), ()),
    ((), (5, 1), (6,), (), ()),
    ((), (), (5, 1), (6,), ()),
    ((), (), (), (5, 1), (6,)),
    ((), (6,), (5, 1), (), ()),
    ((), (), (6,), (5, 1), ()),
    ((), (), (), (6,), (5, 1)),
    ((5,), (), (), (), (6,)),
    ((5,), (5, 6), (5, 6), (5, 6), (5, 6)),
], ids=str)
@pytest.mark.parametrize("diag", [False, True], ids=["full", "diag"])
def test_gaussian_hmm_shape(diag, init_shape, trans_mat_shape, trans_mvn_shape,
                            obs_mat_shape, obs_mvn_shape, hidden_dim, obs_dim):
    init_dist = random_mvn(init_shape, hidden_dim)
    trans_mat = torch.randn(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_mvn(trans_mvn_shape, hidden_dim)
    obs_mat = torch.randn(obs_mat_shape + (hidden_dim, obs_dim))
    obs_dist = random_mvn(obs_mvn_shape, obs_dim)
    if diag:
        scale = obs_dist.scale_tril.diagonal(dim1=-2, dim2=-1)
        obs_dist = dist.Normal(obs_dist.loc, scale).to_event(1)
    d = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                         duration=6)

    shape = broadcast_shape(init_shape + (6,),
                            trans_mat_shape,
                            trans_mvn_shape,
                            obs_mat_shape,
                            obs_mvn_shape)
    expected_batch_shape, time_shape = shape[:-1], shape[-1:]
    expected_event_shape = time_shape + (obs_dim,)
    assert d.batch_shape == expected_batch_shape
    assert d.event_shape == expected_event_shape

    data = obs_dist.expand(shape).sample()
    assert data.shape == d.shape()
    actual = d.log_prob(data)
    assert actual.shape == expected_batch_shape
    check_expand(d, data)

    x = d.rsample()
    assert x.shape == d.shape()
    x = d.rsample((6,))
    assert x.shape == (6,) + d.shape()
    x = d.expand((6, 5)).rsample()
    assert x.shape == (6, 5) + d.event_shape

    likelihood = dist.Normal(data, 1).to_event(2)
    p, log_normalizer = d.conjugate_update(likelihood)
    assert p.batch_shape == d.batch_shape
    assert p.event_shape == d.event_shape
    x = p.rsample()
    assert x.shape == d.shape()
    x = p.rsample((6,))
    assert x.shape == (6,) + d.shape()
    x = p.expand((6, 5)).rsample()
    assert x.shape == (6, 5) + d.event_shape

    final = d.filter(data)
    assert isinstance(final, dist.MultivariateNormal)
    assert final.batch_shape == d.batch_shape
    assert final.event_shape == (hidden_dim,)

    z = d.rsample_posterior(data)
    assert z.shape == expected_batch_shape + time_shape + (hidden_dim,)

    for t in range(1, d.duration - 1):
        f = d.duration - t
        d2 = d.prefix_condition(data[..., :t, :])
        assert d2.batch_shape == d.batch_shape
        assert d2.event_shape == (f, obs_dim)


def test_gaussian_hmm_high_obs_dim():
    hidden_dim = 1
    obs_dim = 1000
    duration = 10
    sample_shape = (100,)
    init_dist = random_mvn((), hidden_dim)
    trans_mat = torch.randn((duration,) + (hidden_dim, hidden_dim))
    trans_dist = random_mvn((duration,), hidden_dim)
    obs_mat = torch.randn((duration,) + (hidden_dim, obs_dim))
    loc = torch.randn((duration, obs_dim))
    scale = torch.randn((duration, obs_dim)).exp()
    obs_dist = dist.Normal(loc, scale).to_event(1)
    d = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                         duration=duration)
    x = d.rsample(sample_shape)
    assert x.shape == sample_shape + (duration, obs_dim)


@pytest.mark.parametrize('sample_shape', [(), (5,)], ids=str)
@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 2])
@pytest.mark.parametrize('num_steps', [1, 2, 3, 4])
@pytest.mark.parametrize("diag", [False, True], ids=["full", "diag"])
def test_gaussian_hmm_distribution(diag, sample_shape, batch_shape, num_steps, hidden_dim, obs_dim):
    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_mat = torch.randn(batch_shape + (num_steps, hidden_dim, hidden_dim))
    trans_dist = random_mvn(batch_shape + (num_steps,), hidden_dim)
    obs_mat = torch.randn(batch_shape + (num_steps, hidden_dim, obs_dim))
    obs_dist = random_mvn(batch_shape + (num_steps,), obs_dim)
    if diag:
        scale = obs_dist.scale_tril.diagonal(dim1=-2, dim2=-1)
        obs_dist = dist.Normal(obs_dist.loc, scale).to_event(1)
    d = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=num_steps)
    if diag:
        obs_mvn = dist.MultivariateNormal(obs_dist.base_dist.loc,
                                          scale_tril=obs_dist.base_dist.scale.diag_embed())
    else:
        obs_mvn = obs_dist
    data = obs_dist.sample(sample_shape)
    assert data.shape == sample_shape + d.shape()
    actual_log_prob = d.log_prob(data)

    # Compare against hand-computed density.
    # We will construct enormous unrolled joint gaussians with shapes:
    #       t | 0 1 2 3 1 2 3      T = 3 in this example
    #   ------+-----------------------------------------
    #    init | H
    #   trans | H H H H            H = hidden
    #     obs |   H H H O O O      O = observed
    #    like |         O O O
    # and then combine these using gaussian_tensordot().
    T = num_steps
    init = mvn_to_gaussian(init_dist)
    trans = matrix_and_mvn_to_gaussian(trans_mat, trans_dist)
    obs = matrix_and_mvn_to_gaussian(obs_mat, obs_mvn)
    like_dist = dist.Normal(torch.randn(data.shape), 1).to_event(2)
    like = mvn_to_gaussian(like_dist)

    unrolled_trans = reduce(operator.add, [
        trans[..., t].event_pad(left=t * hidden_dim, right=(T - t - 1) * hidden_dim)
        for t in range(T)
    ])
    unrolled_obs = reduce(operator.add, [
        obs[..., t].event_pad(left=t * obs.dim(), right=(T - t - 1) * obs.dim())
        for t in range(T)
    ])
    unrolled_like = reduce(operator.add, [
        like[..., t].event_pad(left=t * obs_dim, right=(T - t - 1) * obs_dim)
        for t in range(T)
    ])
    # Permute obs from HOHOHO to HHHOOO.
    perm = torch.cat([torch.arange(hidden_dim) + t * obs.dim() for t in range(T)] +
                     [torch.arange(obs_dim) + hidden_dim + t * obs.dim() for t in range(T)])
    unrolled_obs = unrolled_obs.event_permute(perm)
    unrolled_data = data.reshape(data.shape[:-2] + (T * obs_dim,))

    assert init.dim() == hidden_dim
    assert unrolled_trans.dim() == (1 + T) * hidden_dim
    assert unrolled_obs.dim() == T * (hidden_dim + obs_dim)
    logp = gaussian_tensordot(init, unrolled_trans, hidden_dim)
    logp = gaussian_tensordot(logp, unrolled_obs, T * hidden_dim)
    expected_log_prob = logp.log_density(unrolled_data)
    assert_close(actual_log_prob, expected_log_prob)

    d_posterior, log_normalizer = d.conjugate_update(like_dist)
    assert_close(d.log_prob(data) + like_dist.log_prob(data),
                 d_posterior.log_prob(data) + log_normalizer)

    if batch_shape or sample_shape:
        return

    # Test mean and covariance.
    prior = "prior", d, logp
    posterior = "posterior", d_posterior, logp + unrolled_like
    for name, d, g in [prior, posterior]:
        logging.info("testing {} moments".format(name))
        with torch.no_grad():
            num_samples = 100000
            samples = d.sample([num_samples]).reshape(num_samples, T * obs_dim)
            actual_mean = samples.mean(0)
            delta = samples - actual_mean
            actual_cov = (delta.unsqueeze(-1) * delta.unsqueeze(-2)).mean(0)
            actual_std = actual_cov.diagonal(dim1=-2, dim2=-1).sqrt()
            actual_corr = actual_cov / (actual_std.unsqueeze(-1) * actual_std.unsqueeze(-2))

            expected_cov = g.precision.cholesky().cholesky_inverse()
            expected_mean = expected_cov.matmul(g.info_vec.unsqueeze(-1)).squeeze(-1)
            expected_std = expected_cov.diagonal(dim1=-2, dim2=-1).sqrt()
            expected_corr = expected_cov / (expected_std.unsqueeze(-1) * expected_std.unsqueeze(-2))

            assert_close(actual_mean, expected_mean, atol=0.05, rtol=0.02)
            assert_close(actual_std, expected_std, atol=0.05, rtol=0.02)
            assert_close(actual_corr, expected_corr, atol=0.02)


@pytest.mark.parametrize('obs_dim', [1, 2, 3])
@pytest.mark.parametrize('hidden_dim', [1, 2, 3])
@pytest.mark.parametrize('init_shape,trans_shape,obs_shape', [
    ((), (7,), ()),
    ((), (), (7,)),
    ((), (7,), (1,)),
    ((), (1,), (7,)),
    ((), (7,), (11, 7)),
    ((), (11, 7), (7,)),
    ((), (11, 7), (11, 7)),
    ((11,), (7,), (7,)),
    ((11,), (7,), (11, 7)),
    ((11,), (11, 7), (7,)),
    ((11,), (11, 7), (11, 7)),
    ((4, 1, 1), (3, 1, 7), (2, 7)),
], ids=str)
def test_gaussian_mrf_shape(init_shape, trans_shape, obs_shape, hidden_dim, obs_dim):
    init_dist = random_mvn(init_shape, hidden_dim)
    trans_dist = random_mvn(trans_shape, hidden_dim + hidden_dim)
    obs_dist = random_mvn(obs_shape, hidden_dim + obs_dim)
    d = dist.GaussianMRF(init_dist, trans_dist, obs_dist)

    shape = broadcast_shape(init_shape + (1,), trans_shape, obs_shape)
    expected_batch_shape, time_shape = shape[:-1], shape[-1:]
    expected_event_shape = time_shape + (obs_dim,)
    assert d.batch_shape == expected_batch_shape
    assert d.event_shape == expected_event_shape

    data = obs_dist.expand(shape).sample()[..., hidden_dim:]
    assert data.shape == d.shape()
    actual = d.log_prob(data)
    assert actual.shape == expected_batch_shape
    check_expand(d, data)


@pytest.mark.parametrize('sample_shape', [(), (5,)], ids=str)
@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 2])
@pytest.mark.parametrize('num_steps', [1, 2, 3, 4])
def test_gaussian_mrf_log_prob(sample_shape, batch_shape, num_steps, hidden_dim, obs_dim):
    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_dist = random_mvn(batch_shape + (num_steps,), hidden_dim + hidden_dim)
    obs_dist = random_mvn(batch_shape + (num_steps,), hidden_dim + obs_dim)
    d = dist.GaussianMRF(init_dist, trans_dist, obs_dist)
    data = obs_dist.sample(sample_shape)[..., hidden_dim:]
    assert data.shape == sample_shape + d.shape()
    actual_log_prob = d.log_prob(data)

    # Compare against hand-computed density.
    # We will construct enormous unrolled joint gaussians with shapes:
    #       t | 0 1 2 3 1 2 3      T = 3 in this example
    #   ------+-----------------------------------------
    #    init | H
    #   trans | H H H H            H = hidden
    #     obs |   H H H O O O      O = observed
    # and then combine these using gaussian_tensordot().
    T = num_steps
    init = mvn_to_gaussian(init_dist)
    trans = mvn_to_gaussian(trans_dist)
    obs = mvn_to_gaussian(obs_dist)

    unrolled_trans = reduce(operator.add, [
        trans[..., t].event_pad(left=t * hidden_dim, right=(T - t - 1) * hidden_dim)
        for t in range(T)
    ])
    unrolled_obs = reduce(operator.add, [
        obs[..., t].event_pad(left=t * obs.dim(), right=(T - t - 1) * obs.dim())
        for t in range(T)
    ])
    # Permute obs from HOHOHO to HHHOOO.
    perm = torch.cat([torch.arange(hidden_dim) + t * obs.dim() for t in range(T)] +
                     [torch.arange(obs_dim) + hidden_dim + t * obs.dim() for t in range(T)])
    unrolled_obs = unrolled_obs.event_permute(perm)
    unrolled_data = data.reshape(data.shape[:-2] + (T * obs_dim,))

    assert init.dim() == hidden_dim
    assert unrolled_trans.dim() == (1 + T) * hidden_dim
    assert unrolled_obs.dim() == T * (hidden_dim + obs_dim)
    logp_h = gaussian_tensordot(init, unrolled_trans, hidden_dim)
    logp_oh = gaussian_tensordot(logp_h, unrolled_obs, T * hidden_dim)
    logp_h += unrolled_obs.marginalize(right=T * obs_dim)
    expected_log_prob = logp_oh.log_density(unrolled_data) - logp_h.event_logsumexp()
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize('sample_shape', [(), (5,)], ids=str)
@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 2])
@pytest.mark.parametrize('num_steps', [1, 2, 3, 4])
def test_gaussian_mrf_log_prob_block_diag(sample_shape, batch_shape, num_steps, hidden_dim, obs_dim):
    # Construct a block-diagonal obs dist, so observations are independent of hidden state.
    obs_dist = random_mvn(batch_shape + (num_steps,), hidden_dim + obs_dim)
    precision = obs_dist.precision_matrix
    precision[..., :hidden_dim, hidden_dim:] = 0
    precision[..., hidden_dim:, :hidden_dim] = 0
    obs_dist = dist.MultivariateNormal(obs_dist.loc, precision_matrix=precision)
    marginal_obs_dist = dist.MultivariateNormal(
        obs_dist.loc[..., hidden_dim:],
        precision_matrix=precision[..., hidden_dim:, hidden_dim:])

    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_dist = random_mvn(batch_shape + (num_steps,), hidden_dim + hidden_dim)
    d = dist.GaussianMRF(init_dist, trans_dist, obs_dist)
    data = obs_dist.sample(sample_shape)[..., hidden_dim:]
    assert data.shape == sample_shape + d.shape()
    actual_log_prob = d.log_prob(data)
    expected_log_prob = marginal_obs_dist.log_prob(data).sum(-1)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 3])
@pytest.mark.parametrize('scale_shape,init_shape,trans_mat_shape,trans_mvn_shape,obs_mat_shape,obs_mvn_shape', [
    ((5,), (), (6,), (), (), ()),
    ((), (), (6,), (), (), ()),
    ((), (), (), (6,), (), ()),
    ((), (), (), (), (6,), ()),
    ((), (), (), (), (), (6,)),
    ((), (), (6,), (6,), (6,), (6,)),
    ((), (5,), (6,), (), (), ()),
    ((), (), (5, 1), (6,), (), ()),
    ((), (), (), (5, 1), (6,), ()),
    ((), (), (), (), (5, 1), (6,)),
    ((), (), (6,), (5, 1), (), ()),
    ((), (), (), (6,), (5, 1), ()),
    ((), (), (), (), (6,), (5, 1)),
    ((), (5,), (), (), (), (6,)),
    ((5,), (5,), (5, 6), (5, 6), (5, 6), (5, 6)),
], ids=str)
def test_gamma_gaussian_hmm_shape(scale_shape, init_shape, trans_mat_shape, trans_mvn_shape,
                                  obs_mat_shape, obs_mvn_shape, hidden_dim, obs_dim):
    init_dist = random_mvn(init_shape, hidden_dim)
    trans_mat = torch.randn(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_mvn(trans_mvn_shape, hidden_dim)
    obs_mat = torch.randn(obs_mat_shape + (hidden_dim, obs_dim))
    obs_dist = random_mvn(obs_mvn_shape, obs_dim)
    scale_dist = random_gamma(scale_shape)
    d = dist.GammaGaussianHMM(scale_dist, init_dist, trans_mat, trans_dist, obs_mat, obs_dist)

    shape = broadcast_shape(scale_shape + (1,),
                            init_shape + (1,),
                            trans_mat_shape,
                            trans_mvn_shape,
                            obs_mat_shape,
                            obs_mvn_shape)
    expected_batch_shape, time_shape = shape[:-1], shape[-1:]
    expected_event_shape = time_shape + (obs_dim,)
    assert d.batch_shape == expected_batch_shape
    assert d.event_shape == expected_event_shape

    data = obs_dist.expand(shape).sample()
    assert data.shape == d.shape()
    actual = d.log_prob(data)
    assert actual.shape == expected_batch_shape
    check_expand(d, data)

    mixing, final = d.filter(data)
    assert isinstance(mixing, dist.Gamma)
    assert mixing.batch_shape == d.batch_shape
    assert mixing.event_shape == ()
    assert isinstance(final, dist.MultivariateNormal)
    assert final.batch_shape == d.batch_shape
    assert final.event_shape == (hidden_dim,)


@pytest.mark.parametrize('sample_shape', [(), (5,)], ids=str)
@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 2])
@pytest.mark.parametrize('num_steps', [1, 2, 3, 4])
def test_gamma_gaussian_hmm_log_prob(sample_shape, batch_shape, num_steps, hidden_dim, obs_dim):
    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_mat = torch.randn(batch_shape + (num_steps, hidden_dim, hidden_dim))
    trans_dist = random_mvn(batch_shape + (num_steps,), hidden_dim)
    obs_mat = torch.randn(batch_shape + (num_steps, hidden_dim, obs_dim))
    obs_dist = random_mvn(batch_shape + (num_steps,), obs_dim)
    scale_dist = random_gamma(batch_shape)
    d = dist.GammaGaussianHMM(scale_dist, init_dist, trans_mat, trans_dist, obs_mat, obs_dist)
    obs_mvn = obs_dist
    data = obs_dist.sample(sample_shape)
    assert data.shape == sample_shape + d.shape()
    actual_log_prob = d.log_prob(data)

    # Compare against hand-computed density.
    # We will construct enormous unrolled joint gaussian-gammas with shapes:
    #       t | 0 1 2 3 1 2 3      T = 3 in this example
    #   ------+-----------------------------------------
    #    init | H
    #   trans | H H H H            H = hidden
    #     obs |   H H H O O O      O = observed
    # and then combine these using gamma_gaussian_tensordot().
    T = num_steps
    init = gamma_and_mvn_to_gamma_gaussian(scale_dist, init_dist)
    trans = matrix_and_mvn_to_gamma_gaussian(trans_mat, trans_dist)
    obs = matrix_and_mvn_to_gamma_gaussian(obs_mat, obs_mvn)

    unrolled_trans = reduce(operator.add, [
        trans[..., t].event_pad(left=t * hidden_dim, right=(T - t - 1) * hidden_dim)
        for t in range(T)
    ])
    unrolled_obs = reduce(operator.add, [
        obs[..., t].event_pad(left=t * obs.dim(), right=(T - t - 1) * obs.dim())
        for t in range(T)
    ])
    # Permute obs from HOHOHO to HHHOOO.
    perm = torch.cat([torch.arange(hidden_dim) + t * obs.dim() for t in range(T)] +
                     [torch.arange(obs_dim) + hidden_dim + t * obs.dim() for t in range(T)])
    unrolled_obs = unrolled_obs.event_permute(perm)
    unrolled_data = data.reshape(data.shape[:-2] + (T * obs_dim,))

    assert init.dim() == hidden_dim
    assert unrolled_trans.dim() == (1 + T) * hidden_dim
    assert unrolled_obs.dim() == T * (hidden_dim + obs_dim)
    logp = gamma_gaussian_tensordot(init, unrolled_trans, hidden_dim)
    logp = gamma_gaussian_tensordot(logp, unrolled_obs, T * hidden_dim)
    # compute log_prob of the joint student-t distribution
    expected_log_prob = logp.compound().log_prob(unrolled_data)
    assert_close(actual_log_prob, expected_log_prob)


def random_stable(stability, skew_scale_loc_shape):
    skew = dist.Uniform(-1, 1).sample(skew_scale_loc_shape)
    scale = torch.rand(skew_scale_loc_shape).exp()
    loc = torch.randn(skew_scale_loc_shape)
    return dist.Stable(stability, skew, scale, loc)


@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 3])
@pytest.mark.parametrize('init_shape,trans_mat_shape,trans_dist_shape,obs_mat_shape,obs_dist_shape', [
    ((), (), (), (), ()),
    ((), (4,), (), (), ()),
    ((), (), (4,), (), ()),
    ((), (), (), (4,), ()),
    ((), (), (), (), (4,)),
    ((), (4,), (4,), (4,), (4,)),
    ((5,), (4,), (), (), ()),
    ((), (5, 1), (4,), (), ()),
    ((), (), (5, 1), (4,), ()),
    ((), (), (), (5, 1), (4,)),
    ((), (4,), (5, 1), (), ()),
    ((), (), (4,), (5, 1), ()),
    ((), (), (), (4,), (5, 1)),
    ((5,), (), (), (), (4,)),
    ((5,), (5, 4), (5, 4), (5, 4), (5, 4)),
], ids=str)
def test_stable_hmm_shape(init_shape, trans_mat_shape, trans_dist_shape,
                          obs_mat_shape, obs_dist_shape, hidden_dim, obs_dim):
    stability = dist.Uniform(0, 2).sample()
    init_dist = random_stable(stability, init_shape + (hidden_dim,)).to_event(1)
    trans_mat = torch.randn(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_stable(stability, trans_dist_shape + (hidden_dim,)).to_event(1)
    obs_mat = torch.randn(obs_mat_shape + (hidden_dim, obs_dim))
    obs_dist = random_stable(stability, obs_dist_shape + (obs_dim,)).to_event(1)
    d = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                       duration=4)

    shape = broadcast_shape(init_shape + (4,),
                            trans_mat_shape,
                            trans_dist_shape,
                            obs_mat_shape,
                            obs_dist_shape)
    expected_batch_shape, time_shape = shape[:-1], shape[-1:]
    expected_event_shape = time_shape + (obs_dim,)
    assert d.batch_shape == expected_batch_shape
    assert d.event_shape == expected_event_shape

    x = d.rsample()
    assert x.shape == d.shape()
    x = d.rsample((6,))
    assert x.shape == (6,) + d.shape()
    x = d.expand((6, 5)).rsample()
    assert x.shape == (6, 5) + d.event_shape


def random_studentt(shape):
    df = torch.rand(shape).exp()
    loc = torch.randn(shape)
    scale = torch.rand(shape).exp()
    return dist.StudentT(df, loc, scale)


@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 3])
@pytest.mark.parametrize('init_shape,trans_mat_shape,trans_dist_shape,obs_mat_shape,obs_dist_shape', [
    ((), (4,), (), (), ()),
    ((), (), (4,), (), ()),
    ((), (), (), (4,), ()),
    ((), (), (), (), (4,)),
    ((), (4,), (4,), (4,), (4,)),
    ((5,), (4,), (), (), ()),
    ((), (5, 1), (4,), (), ()),
    ((), (), (5, 1), (4,), ()),
    ((), (), (), (5, 1), (4,)),
    ((), (4,), (5, 1), (), ()),
    ((), (), (4,), (5, 1), ()),
    ((), (), (), (4,), (5, 1)),
    ((5,), (), (), (), (4,)),
    ((5,), (5, 4), (5, 4), (5, 4), (5, 4)),
], ids=str)
def test_studentt_hmm_shape(init_shape, trans_mat_shape, trans_dist_shape,
                            obs_mat_shape, obs_dist_shape, hidden_dim, obs_dim):
    init_dist = random_studentt(init_shape + (hidden_dim,)).to_event(1)
    trans_mat = torch.randn(trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_studentt(trans_dist_shape + (hidden_dim,)).to_event(1)
    obs_mat = torch.randn(obs_mat_shape + (hidden_dim, obs_dim))
    obs_dist = random_studentt(obs_dist_shape + (obs_dim,)).to_event(1)
    d = dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)

    shape = broadcast_shape(init_shape + (1,),
                            trans_mat_shape,
                            trans_dist_shape,
                            obs_mat_shape,
                            obs_dist_shape)
    expected_batch_shape, time_shape = shape[:-1], shape[-1:]
    expected_event_shape = time_shape + (obs_dim,)
    assert d.batch_shape == expected_batch_shape
    assert d.event_shape == expected_event_shape

    x = d.rsample()
    assert x.shape == d.shape()
    x = d.rsample((6,))
    assert x.shape == (6,) + d.shape()
    x = d.expand((6, 5)).rsample()
    assert x.shape == (6, 5) + d.event_shape


@pytest.mark.parametrize('obs_dim', [1, 3])
@pytest.mark.parametrize('hidden_dim', [1, 2])
@pytest.mark.parametrize('init_shape,trans_mat_shape,trans_mvn_shape,obs_mat_shape,obs_mvn_shape', [
    ((), (), (), (), ()),
    ((), (6,), (), (), ()),
    ((), (), (6,), (), ()),
    ((), (), (), (6,), ()),
    ((), (), (), (), (6,)),
    ((), (6,), (6,), (6,), (6,)),
    ((5,), (6,), (), (), ()),
    ((), (5, 1), (6,), (), ()),
    ((), (), (5, 1), (6,), ()),
    ((), (), (), (5, 1), (6,)),
    ((), (6,), (5, 1), (), ()),
    ((), (), (6,), (5, 1), ()),
    ((), (), (), (6,), (5, 1)),
    ((5,), (), (), (), (6,)),
    ((5,), (5, 6), (5, 6), (5, 6), (5, 6)),
], ids=str)
def test_independent_hmm_shape(init_shape, trans_mat_shape, trans_mvn_shape,
                               obs_mat_shape, obs_mvn_shape, hidden_dim, obs_dim):
    base_init_shape = init_shape + (obs_dim,)
    base_trans_mat_shape = trans_mat_shape[:-1] + (obs_dim, trans_mat_shape[-1] if trans_mat_shape else 6)
    base_trans_mvn_shape = trans_mvn_shape[:-1] + (obs_dim, trans_mvn_shape[-1] if trans_mvn_shape else 6)
    base_obs_mat_shape = obs_mat_shape[:-1] + (obs_dim, obs_mat_shape[-1] if obs_mat_shape else 6)
    base_obs_mvn_shape = obs_mvn_shape[:-1] + (obs_dim, obs_mvn_shape[-1] if obs_mvn_shape else 6)

    init_dist = random_mvn(base_init_shape, hidden_dim)
    trans_mat = torch.randn(base_trans_mat_shape + (hidden_dim, hidden_dim))
    trans_dist = random_mvn(base_trans_mvn_shape, hidden_dim)
    obs_mat = torch.randn(base_obs_mat_shape + (hidden_dim, 1))
    obs_dist = random_mvn(base_obs_mvn_shape, 1)
    d = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=6)
    d = dist.IndependentHMM(d)

    shape = broadcast_shape(init_shape + (6,),
                            trans_mat_shape,
                            trans_mvn_shape,
                            obs_mat_shape,
                            obs_mvn_shape)
    expected_batch_shape, time_shape = shape[:-1], shape[-1:]
    expected_event_shape = time_shape + (obs_dim,)
    assert d.batch_shape == expected_batch_shape
    assert d.event_shape == expected_event_shape

    data = torch.randn(shape + (obs_dim,))
    assert data.shape == d.shape()
    actual = d.log_prob(data)
    assert actual.shape == expected_batch_shape
    check_expand(d, data)

    x = d.rsample()
    assert x.shape == d.shape()
    x = d.rsample((6,))
    assert x.shape == (6,) + d.shape()
    x = d.expand((6, 5)).rsample()
    assert x.shape == (6, 5) + d.event_shape
