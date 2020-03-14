# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.distributions.transforms import ComposeTransform
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoIAFNormal
from pyro.infer.mcmc.util import initialize_model
from pyro.infer.reparam import NeuTraReparam
from tests.common import assert_close, xfail_param


def neals_funnel(dim):
    y = pyro.sample('y', dist.Normal(0, 3))
    with pyro.plate('D', dim):
        pyro.sample('x', dist.Normal(0, torch.exp(y / 2)))


def dirichlet_categorical(data):
    concentration = torch.tensor([1.0, 1.0, 1.0])
    p_latent = pyro.sample('p', dist.Dirichlet(concentration))
    with pyro.plate('N', data.shape[0]):
        pyro.sample('obs', dist.Categorical(p_latent), obs=data)
    return p_latent


@pytest.mark.parametrize('jit', [
    False,
    xfail_param(True, reason="https://github.com/pyro-ppl/pyro/issues/2292"),
])
def test_neals_funnel_smoke(jit):
    dim = 10

    guide = AutoIAFNormal(neals_funnel)
    svi = SVI(neals_funnel, guide,  optim.Adam({"lr": 1e-10}), Trace_ELBO())
    for _ in range(1000):
        svi.step(dim)

    neutra = NeuTraReparam(guide.requires_grad_(False))
    model = neutra.reparam(neals_funnel)
    nuts = NUTS(model, jit_compile=jit)
    mcmc = MCMC(nuts, num_samples=50, warmup_steps=50)
    mcmc.run(dim)
    samples = mcmc.get_samples()
    # XXX: `MCMC.get_samples` adds a leftmost batch dim to all sites, not uniformly at -max_plate_nesting-1;
    # hence the unsqueeze
    transformed_samples = neutra.transform_sample(samples['y_shared_latent'].unsqueeze(-2))
    assert 'x' in transformed_samples
    assert 'y' in transformed_samples


@pytest.mark.parametrize('model, kwargs', [
    (neals_funnel, {'dim': 10}),
    (dirichlet_categorical, {'data': torch.ones(10,)})
])
def test_reparam_log_joint(model, kwargs):
    guide = AutoIAFNormal(model)
    guide(**kwargs)
    neutra = NeuTraReparam(guide)
    reparam_model = neutra.reparam(model)
    _, pe_fn, transforms, _ = initialize_model(model, model_kwargs=kwargs)
    init_params, pe_fn_neutra, _, _ = initialize_model(reparam_model, model_kwargs=kwargs)
    latent_x = list(init_params.values())[0]
    transformed_params = neutra.transform_sample(latent_x)
    pe_transformed = pe_fn_neutra(init_params)
    neutra_transform = ComposeTransform(guide.get_posterior(**kwargs).transforms)
    latent_y = neutra_transform(latent_x)
    log_det_jacobian = neutra_transform.log_abs_det_jacobian(latent_x, latent_y)
    pe = pe_fn({k: transforms[k](v) for k, v in transformed_params.items()})
    assert_close(pe_transformed, pe - log_det_jacobian)
