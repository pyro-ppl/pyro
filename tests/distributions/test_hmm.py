from __future__ import absolute_import, division, print_function

import opt_einsum
import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.hmm import _sequential_logmatmulexp
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
