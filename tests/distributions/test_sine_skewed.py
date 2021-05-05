from math import pi

import pytest
import torch
from numpy.testing import assert_raises

import pyro
from pyro.distributions import Normal, SineSkewed, Uniform, constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tests.common import assert_equal

BASE_DISTS = [(Uniform, ([-pi, -pi], [pi, pi]))]


def _skewness(event_shape):
    skewness = torch.zeros(event_shape.numel())
    done = False
    while not done:
        for i in range(event_shape.numel()):
            max_ = 1. - skewness.abs().sum(-1)
            if torch.any(max_ < 1e-15):
                break
            skewness[i] = Uniform(-max_, max_).sample()
        done = not torch.any(max_ < 1e-15)

    if event_shape == tuple():
        skewness = skewness.reshape(event_shape)
    else:
        skewness = skewness.view(event_shape)
    return skewness


@pytest.mark.parametrize('expand_shape',
                         [(), (1,), (4,), (1, 1), (1, 2), (10, 10), (1, 3, 1), (10, 1, 5), (1, 1, 1), (3, 2, 3)])
@pytest.mark.parametrize('event_dim', [0, 1])
@pytest.mark.parametrize('dist', BASE_DISTS)
def test_ss_multidim_log_prob(event_dim, expand_shape, dist):
    if len(expand_shape) >= event_dim and event_dim:
        base_dist = dist[0](*(torch.tensor(param).expand(*expand_shape, 2) for param in dist[1]))
        base_dist = base_dist.to_event(event_dim + 1)
        assert base_dist.batch_shape == expand_shape[:-event_dim]
    else:
        base_dist = dist[0](*(torch.tensor(param) for param in dist[1])).to_event(1)
        base_dist = base_dist.expand(expand_shape)
        assert base_dist.batch_shape == expand_shape

    loc = Normal(0., 1.).sample(base_dist.event_shape) % (2 * pi) - pi

    base_prob = base_dist.log_prob(loc)
    skewness = _skewness(base_dist.event_shape)

    ss = SineSkewed(base_dist, skewness)
    assert_equal(base_prob + torch.log(
        1 + (skewness * torch.sin(loc - base_dist.mean)).view(*base_dist.batch_shape, -1).sum(-1)),
                 ss.log_prob(loc))
    assert_equal(ss.sample().shape, torch.Size((*expand_shape, 2)))


def test_ss_invalid_event_shape():
    base_dist = Uniform(-1, 1).expand((3, 3, 2)).to_event(3)
    assert_raises(AssertionError, SineSkewed, base_dist, torch.zeros(base_dist.shape()))
    base_dist = Uniform(-1, 1).expand((5,)).to_event(1)
    assert_raises(AssertionError, SineSkewed, base_dist, torch.zeros(base_dist.shape()))


def test_ss_skewness_too_high():
    base_dist = Uniform(-1, 1).expand((2,)).to_event(1)
    assert_raises(AssertionError, SineSkewed, base_dist, torch.ones(base_dist.shape()))
    base_dist = Uniform(-1, 1).expand((1, 2,)).to_event(1)
    assert_raises(AssertionError, SineSkewed, base_dist, .51 * torch.ones(base_dist.shape()))
    base_dist = Uniform(-1, 1).expand((2, 2,)).to_event(1)
    assert_raises(AssertionError, SineSkewed, base_dist, .5 * torch.ones(base_dist.shape()))


@pytest.mark.parametrize('dist', BASE_DISTS)
def test_ss_sample(dist):
    base_dist = dist[0](*(torch.tensor(param) for param in dist[1])).to_event(1)

    skewness_tar = _skewness(base_dist.event_shape)
    data = SineSkewed(base_dist, skewness_tar).sample((1000,))

    def model(data, batch_shape):
        skew0 = pyro.param('skew0', torch.zeros(batch_shape), constraint=constraints.interval(-1, 1))
        skew1 = pyro.param('skew1', torch.zeros(batch_shape), constraint=constraints.interval(-1, 1))

        skewness = torch.stack((skew0, skew1), dim=-1)
        with pyro.plate("data", data.size(-len(data.size()))):
            pyro.sample('obs', SineSkewed(base_dist, skewness), obs=data)

    def guide(data, batch_shape):
        pass

    pyro.clear_param_store()
    adam = Adam({"lr": .1})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    losses = []
    steps = 80
    for step in range(steps):
        losses.append(svi.step(data, base_dist.batch_shape))

    act_skewness = torch.stack([v for k, v in pyro.get_param_store().items() if 'skew' in k], dim=-1)
    assert_equal(act_skewness, skewness_tar, 5e-2)
