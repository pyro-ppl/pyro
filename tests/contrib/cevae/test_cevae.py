import io

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.cevae import CEVAE
from tests.common import assert_close


def generate_data(num_data, feature_dim):
    z = dist.Bernoulli(0.5).sample([num_data])
    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([feature_dim]).t()
    t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
    y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()
    return x, t, y


@pytest.mark.parametrize("num_data", [1, 100, 200])
@pytest.mark.parametrize("feature_dim", [1, 2])
def test_smoke(num_data, feature_dim):
    x, t, y = generate_data(num_data, feature_dim)
    cevae = CEVAE(feature_dim)
    cevae.fit(x, t, y, num_epochs=2)
    ite = cevae.ite(x)
    assert ite.shape == (num_data,)


@pytest.mark.parametrize("feature_dim", [1, 2])
def test_serialization(feature_dim):
    x, t, y = generate_data(num_data=32, feature_dim=feature_dim)
    cevae = CEVAE(feature_dim, num_samples=1000)
    cevae.fit(x, t, y, num_epochs=4, batch_size=8)
    pyro.set_rng_seed(0)
    expected_ite = cevae.ite(x)

    # Ignore tracer warnings
    traced_cevae = cevae.jit_trace()
    f = io.BytesIO()
    torch.jit.save(traced_cevae, f)
    f.seek(0)
    loaded_cevae = torch.jit.load(f)

    # Check .call() result.
    pyro.set_rng_seed(0)
    actual_ite = loaded_cevae.ite(x)
    assert_close(actual_ite, expected_ite, atol=0.1)
