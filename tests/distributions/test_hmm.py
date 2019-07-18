import opt_einsum
import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.hmm import _sequential_logmatmulexp
from pyro.distributions.util import broadcast_shape
from pyro.infer import TraceEnum_ELBO, config_enumerate
from pyro.ops.indexing import Vindex
from tests.common import assert_close


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
def test_shape(ok, init_shape, trans_shape, obs_shape, event_shape, state_dim):
    init_logits = torch.randn(init_shape + (state_dim,))
    trans_logits = torch.randn(trans_shape + (state_dim, state_dim))
    obs_logits = torch.randn(obs_shape + (state_dim,) + event_shape)
    obs_dist = dist.Bernoulli(logits=obs_logits).to_event(len(event_shape))
    data = obs_dist.sample()[(slice(None),) * len(obs_shape) + (0,)]

    if ok:
        d = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
        actual = d.log_prob(data)
        expected_shape = broadcast_shape(init_shape, trans_shape[:-1], obs_shape[:-1])
        assert actual.shape == expected_shape
    else:
        with pytest.raises(ValueError):
            dist.DiscreteHMM(init_logits, trans_logits, obs_dist)


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
def test_homogeneous_homogeneous_trick(init_shape, trans_shape, obs_shape, event_shape, state_dim, num_steps):
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
def test_categorical(num_steps):
    state_dim = 3
    obs_dim = 4
    init_logits = torch.randn(state_dim)
    trans_logits = torch.randn(num_steps, state_dim, state_dim)
    obs_dist = dist.Categorical(logits=torch.randn(num_steps, state_dim, obs_dim))
    d = dist.DiscreteHMM(init_logits, trans_logits, obs_dist)
    data = dist.Categorical(logits=torch.zeros(num_steps, obs_dim)).sample()
    actual = d.log_prob(data)
    assert actual.shape == d.batch_shape

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
def test_diag_normal(num_steps):
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
