import pytest

from pyro.generic import infer, pyro_backend, handlers
from pyro.generic.testing import MODELS


pytestmark = pytest.mark.stage('unit')


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', ['pyro'])
def test_mcmc_interface(model, backend):
    with pyro_backend(backend), handlers.seed(rng_seed=20):
        f = MODELS[model]()
        model, args, kwargs = f['model'], f.get('model_args', ()), f.get('model_kwargs', {})
        nuts_kernel = infer.NUTS(model=model)
        mcmc = infer.MCMC(nuts_kernel, num_samples=10, warmup_steps=10)
        mcmc.run(*args, **kwargs)
        mcmc.summary()
