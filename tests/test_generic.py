# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyro.generic import handlers, infer, pyro, pyro_backend, ops
from pyroapi.testing import MODELS
from tests.common import xfail_if_not_implemented

pytestmark = pytest.mark.stage('unit')


@pytest.mark.filterwarnings("ignore", category=UserWarning)
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


@pytest.mark.parametrize('backend', ['pyro', 'minipyro'])
def test_not_implemented(backend):
    with pyro_backend(backend):
        pyro.sample  # should be implemented
        pyro.param  # should be implemented
        with pytest.raises(NotImplementedError):
            pyro.nonexistent_primitive


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', ['minipyro', 'pyro'])
def test_model_sample(model, backend):
    with pyro_backend(backend), handlers.seed(rng_seed=2), xfail_if_not_implemented():
        f = MODELS[model]()
        model, model_args = f['model'], f.get('model_args', ())
        model(*model_args)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', ['minipyro', 'pyro'])
def test_rng_seed(model, backend):
    with pyro_backend(backend), handlers.seed(rng_seed=2), xfail_if_not_implemented():
        f = MODELS[model]()
        model, model_args = f['model'], f.get('model_args', ())
        with handlers.seed(rng_seed=0):
            expected = model(*model_args)
        if expected is None:
            pytest.skip()
        with handlers.seed(rng_seed=0):
            actual = model(*model_args)
        assert ops.allclose(actual, expected)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', ['minipyro', 'pyro'])
def test_rng_state(model, backend):
    with pyro_backend(backend), handlers.seed(rng_seed=2), xfail_if_not_implemented():
        f = MODELS[model]()
        model, model_args = f['model'], f.get('model_args', ())
        with handlers.seed(rng_seed=0):
            model(*model_args)
            expected = model(*model_args)
        if expected is None:
            pytest.skip()
        with handlers.seed(rng_seed=0):
            model(*model_args)
            with handlers.seed(rng_seed=0):
                model(*model_args)
            actual = model(*model_args)
        assert ops.allclose(actual, expected)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('backend', ['minipyro', 'pyro'])
def test_trace_handler(model, backend):
    with pyro_backend(backend), handlers.seed(rng_seed=2), xfail_if_not_implemented():
        f = MODELS[model]()
        model, model_args, model_kwargs = f['model'], f.get('model_args', ()), f.get('model_kwargs', {})
        # should be implemented
        handlers.trace(model).get_trace(*model_args, **model_kwargs)
