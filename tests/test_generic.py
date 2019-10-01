import pytest

from pyro.generic import infer, pyro_backend, seed
from pyro.generic.testing import MODELS


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', ['pyro', 'numpy'])
def test_mcmc_interface(model, backend):
    with pyro_backend(backend), seed(rng_key=20):
        f = MODELS[model]()
        model, args, kwargs = f['model'], f.get('model_args', ()), f.get('model_kwargs', {})
        nuts_kernel = infer.NUTS(model=model)
        mcmc = infer.MCMC(nuts_kernel, num_samples=10, warmup_steps=10)
        mcmc.run(*args, **kwargs)
        mcmc.summary()
