# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from math import pi

import pytest
import torch

import pyro
from pyro.distributions import Normal, SineSkewed, Uniform, VonMises, constraints
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tests.common import assert_equal

BASE_DISTS = [(Uniform, [-pi, pi]), (VonMises, (0.0, 1.0))]


def _skewness(event_shape):
    skewness = torch.zeros(event_shape.numel())
    done = False
    while not done:
        for i in range(event_shape.numel()):
            max_ = 1.0 - skewness.abs().sum(-1)
            if torch.any(max_ < 1e-15):
                break
            skewness[i] = Uniform(-max_, max_).sample()
        done = not torch.any(max_ < 1e-15)

    if event_shape == tuple():
        skewness = skewness.reshape(event_shape)
    else:
        skewness = skewness.view(event_shape)
    return skewness


@pytest.mark.parametrize(
    "expand_shape",
    [
        (1,),
        (2,),
        (4,),
        (1, 1),
        (1, 2),
        (10, 10),
        (1, 3, 1),
        (10, 1, 5),
        (1, 1, 1),
        (3, 2, 3),
    ],
)
@pytest.mark.parametrize("dist", BASE_DISTS)
def test_ss_multidim_log_prob(expand_shape, dist):
    base_dist = dist[0](
        *(torch.tensor(param).expand(expand_shape) for param in dist[1])
    ).to_event(1)

    loc = base_dist.sample((10,)) + Normal(0.0, 1e-3).sample()

    base_prob = base_dist.log_prob(loc)
    skewness = _skewness(base_dist.event_shape)

    ss = SineSkewed(base_dist, skewness)
    assert_equal(base_prob.shape, ss.log_prob(loc).shape)
    assert_equal(ss.sample().shape, torch.Size(expand_shape))


@pytest.mark.parametrize("dist", BASE_DISTS)
@pytest.mark.parametrize("dim", [1, 2])
def test_ss_mle(dim, dist):
    base_dist = dist[0](
        *(torch.tensor(param).expand((dim,)) for param in dist[1])
    ).to_event(1)

    skewness_tar = _skewness(base_dist.event_shape)
    data = SineSkewed(base_dist, skewness_tar).sample((1000,))

    def model(data, batch_shape):
        skews = []
        for i in range(dim):
            skews.append(
                pyro.param(
                    f"skew{i}",
                    0.5 * torch.ones(batch_shape),
                    constraint=constraints.interval(-1, 1),
                )
            )

        skewness = torch.stack(skews, dim=-1)
        with pyro.plate("data", data.size(-len(data.size()))):
            pyro.sample("obs", SineSkewed(base_dist, skewness), obs=data)

    def guide(data, batch_shape):
        pass

    pyro.clear_param_store()
    adam = Adam({"lr": 0.1})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    losses = []
    steps = 80
    for step in range(steps):
        losses.append(svi.step(data, base_dist.batch_shape))

    act_skewness = torch.stack(
        [v for k, v in pyro.get_param_store().items() if "skew" in k], dim=-1
    )
    assert_equal(act_skewness, skewness_tar, 1e-1)
