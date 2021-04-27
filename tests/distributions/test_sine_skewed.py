from math import pi

import pytest
import torch

import pyro
from pyro.distributions import Uniform, Normal
from pyro.distributions.sine_skewed import SineSkewed
from pyro.infer import Trace_ELBO, SVI
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from tests.common import assert_equal

BASE_DISTS = [(Uniform, ([-pi, -pi], [pi, pi]))]


def _skewness(batch_shape, event_shape):
    n = torch.prod(torch.tensor(event_shape), -1, dtype=torch.int)
    skewness = torch.empty((*batch_shape, n)).view(-1, n)
    tots = torch.zeros(batch_shape).view(-1)
    for i in range(n):
        skewness[..., i] = Uniform(0., 1 - tots).sample()
        tots += skewness[..., i]
    skewness = torch.where(Uniform(0, 1.).sample(skewness.shape) < .5, -skewness, skewness)
    if (*batch_shape, *event_shape) == tuple():
        skewness = skewness.reshape((*batch_shape, *event_shape))
    else:
        skewness = skewness.view(*batch_shape, *event_shape)
    return skewness


@pytest.mark.parametrize('dist', BASE_DISTS)
def test_ss_log_prob(dist):
    base_dist = dist[0](*(torch.tensor(param) for param in dist[1])).to_event(1)
    loc = Normal(0., 1.).sample(base_dist.mean.shape) % (2 * pi) - pi

    base_prob = base_dist.log_prob(loc)
    skewness = _skewness(base_dist.batch_shape, base_dist.event_shape)
    sine_prob = SineSkewed(base_dist, skewness).log_prob(loc)
    assert_equal(base_prob + torch.log(1 + (skewness * torch.sin(loc - base_dist.mean)).sum()), sine_prob)


@pytest.mark.parametrize('dist', BASE_DISTS)
def test_ss_sample(dist):
    base_dist = dist[0](*(torch.tensor(param) for param in dist[1])).to_event(1)

    skewness_tar = _skewness(base_dist.batch_shape, base_dist.event_shape)
    data = SineSkewed(base_dist, skewness_tar).sample((1000,))

    def model(data, batch_shape, event_shape):
        n = torch.prod(torch.tensor(event_shape), -1, dtype=torch.int)
        skewness = torch.empty((*batch_shape, n)).view(-1, n)
        tots = torch.zeros(batch_shape).view(-1)
        for i in range(n):
            skewness[..., i] = pyro.sample(f'skew{i}', Uniform(0., 1 - tots))
            tots += skewness[..., i]
        sign = pyro.sample('sign', Uniform(0., torch.ones(skewness.shape)).to_event(len(skewness.shape)))
        skewness = torch.where(sign < .5, -skewness, skewness)

        if (*batch_shape, *event_shape) == tuple():
            skewness = skewness.reshape((*batch_shape, *event_shape))
        else:
            skewness = skewness.view(*batch_shape, *event_shape)

        with pyro.plate("data", data.size(-len(data.size()))):
            pyro.sample('obs', SineSkewed(base_dist, skewness), obs=data)

    pyro.clear_param_store()
    adam = Adam({"lr": .1})
    guide = AutoDelta(model)
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    losses = []
    steps = 50
    for step in range(steps):
        losses.append(svi.step(data, base_dist.batch_shape, base_dist.event_shape))

    act_sign = pyro.param('AutoDelta.sign')
    act_skewness = torch.stack([v for k, v in pyro.get_param_store().items() if 'skew' in k]).T
    act_skewness = torch.where(act_sign < .5, -act_skewness, act_skewness)

    if (*base_dist.batch_shape, *base_dist.event_shape) == tuple():
        act_skewness = act_skewness.reshape((*base_dist.batch_shape, *base_dist.event_shape))
    else:
        act_skewness = act_skewness.view(*base_dist.batch_shape, *base_dist.event_shape)

    assert_equal(act_skewness, skewness_tar, 5e-2)
