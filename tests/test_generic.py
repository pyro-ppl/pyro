import pytest
import torch

from pyro.generic import handlers, infer, pyro, pyro_backend
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
        if torch.backends.mkl.is_available():
            mcmc.summary()


@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_not_implemented(backend):
    with pyro_backend(backend):
        pyro.sample  # should be implemented
        pyro.param  # should be implemented
        with pytest.raises(NotImplementedError):
            pyro.nonexistant_primitive
