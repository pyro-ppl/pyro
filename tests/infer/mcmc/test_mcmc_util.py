import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.util import initialize_model, predictive
from pyro.util import ignore_experimental_warning
from tests.common import assert_close


def beta_bernoulli():
    N = 1000
    true_probs = torch.tensor([0.2, 0.3, 0.4, 0.8, 0.5])
    data = dist.Bernoulli(true_probs).sample([N])

    def model(data=None):
        with pyro.plate("num_components", 5):
            beta = pyro.sample("beta", dist.Beta(1., 1.))
            with pyro.plate("data", N):
                pyro.sample("obs", dist.Bernoulli(beta), obs=data)

    return model, data, true_probs


def test_predictive():
    model, data, true_probs = beta_bernoulli()
    init_params, potential_fn, transforms, _ = initialize_model(model,
                                                                model_args=(data,))
    nuts_kernel = NUTS(potential_fn=potential_fn, transforms=transforms)
    samples = MCMC(nuts_kernel,
                   100,
                   initial_params=init_params,
                   warmup_steps=100).run(data)
    with ignore_experimental_warning():
        predictive_samples = predictive(model, samples,
                                        return_sites=["beta", "obs"],
                                        num_samples=200)
    assert_close(predictive_samples["obs"].reshape([-1, 5]).mean(0), true_probs, rtol=0.1)
