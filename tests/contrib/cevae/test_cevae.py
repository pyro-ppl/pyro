import pytest

import pyro.distributions as dist
from pyro.contrib.cevae import CEVAE


@pytest.mark.parametrize("num_data", [1, 100, 200])
@pytest.mark.parametrize("feature_dim", [1, 2])
def test_smoke(num_data, feature_dim):
    z = dist.Bernoulli(0.5).sample([num_data])
    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([feature_dim]).t()
    t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
    y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()

    cevae = CEVAE(feature_dim)
    cevae.fit(x, t, y, num_epochs=2)
    ite = cevae.ite(x)
    assert ite.shape == (num_data,)
