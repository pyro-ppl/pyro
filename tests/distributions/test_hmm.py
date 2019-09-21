import operator
from functools import reduce

import opt_einsum
import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.hmm import _sequential_gaussian_tensordot, _sequential_logmatmulexp
from pyro.distributions.util import broadcast_shape
from pyro.infer import TraceEnum_ELBO, config_enumerate
from pyro.ops.gaussian import gaussian_tensordot, matrix_and_mvn_to_gaussian, mvn_to_gaussian
from pyro.ops.indexing import Vindex
from tests.common import assert_close
from tests.ops.gaussian import assert_close_gaussian, random_gaussian, random_mvn


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


@pytest.mark.parametrize('state_dim', [2, 3])
@pytest.mark.parametrize('event_shape', [(), (5,), (2, 3)], ids=str)
@pytest.mark.parametrize('ok,init_shape,trans_shape,obs_shape', [
    (True, (), (1,), ()),
    (True, (), (), (1,)),
    (True, (), (7,), ()),
    (True, (), (), (7,)),
    (True, (), (7,), (1,)),
    (True, (), (1,), (7,)),
    (True, (), (7,), (11, 7)),
    (True, (), (11, 7), (7,)),
    (True, (), (11, 7), (11, 7)),
    (True, (11,), (7,), (7,)),
    (True, (11,), (7,), (11, 7)),
    (True, (11,), (11, 7), (7,)),
    (True, (11,), (11, 7), (11, 7)),
    (True, (4, 1, 1), (3, 1, 7), (2, 7)),
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
            dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
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
    d = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)

    shape = broadcast_shape(init_shape + (1,),
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

    final = d.filter(data)
    assert isinstance(final, dist.MultivariateNormal)
    assert final.batch_shape == d.batch_shape
    assert final.event_shape == (hidden_dim,)


@pytest.mark.parametrize('sample_shape', [(), (5,)], ids=str)
@pytest.mark.parametrize('batch_shape', [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize('obs_dim', [1, 2])
@pytest.mark.parametrize('hidden_dim', [1, 2])
@pytest.mark.parametrize('num_steps', [1, 2, 3, 4])
@pytest.mark.parametrize("diag", [False, True], ids=["full", "diag"])
def test_gaussian_hmm_log_prob(diag, sample_shape, batch_shape, num_steps, hidden_dim, obs_dim):
    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_mat = torch.randn(batch_shape + (num_steps, hidden_dim, hidden_dim))
    trans_dist = random_mvn(batch_shape + (num_steps,), hidden_dim)
    obs_mat = torch.randn(batch_shape + (num_steps, hidden_dim, obs_dim))
    obs_dist = random_mvn(batch_shape + (num_steps,), obs_dim)
    if diag:
        scale = obs_dist.scale_tril.diagonal(dim1=-2, dim2=-1)
        obs_dist = dist.Normal(obs_dist.loc, scale).to_event(1)
    d = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)
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
    # and then combine these using gaussian_tensordot().
    T = num_steps
    init = mvn_to_gaussian(init_dist)
    trans = matrix_and_mvn_to_gaussian(trans_mat, trans_dist)
    obs = matrix_and_mvn_to_gaussian(obs_mat, obs_mvn)

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
    logp = gaussian_tensordot(init, unrolled_trans, hidden_dim)
    logp = gaussian_tensordot(logp, unrolled_obs, T * hidden_dim)
    expected_log_prob = logp.log_density(unrolled_data)
    assert_close(actual_log_prob, expected_log_prob)


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
