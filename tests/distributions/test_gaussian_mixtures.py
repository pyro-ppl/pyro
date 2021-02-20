# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import torch

import pytest
from pyro.distributions import MixtureOfDiagNormalsSharedCovariance, GaussianScaleMixture
from pyro.distributions import MixtureOfDiagNormals
from tests.common import assert_equal


logger = logging.getLogger(__name__)


@pytest.mark.parametrize('mix_dist', [MixtureOfDiagNormals, MixtureOfDiagNormalsSharedCovariance, GaussianScaleMixture])
@pytest.mark.parametrize('K', [3])
@pytest.mark.parametrize('D', [2, 4])
@pytest.mark.parametrize('batch_mode', [True, False])
@pytest.mark.parametrize('flat_logits', [True, False])
@pytest.mark.parametrize('cost_function', ['quadratic'])
def test_mean_gradient(K, D, flat_logits, cost_function, mix_dist, batch_mode):
    n_samples = 200000
    if batch_mode:
        sample_shape = torch.Size(())
    else:
        sample_shape = torch.Size((n_samples,))
    if mix_dist == GaussianScaleMixture:
        locs = torch.zeros(K, D, requires_grad=True)
    else:
        locs = torch.rand(K, D).requires_grad_(True)
    if mix_dist == GaussianScaleMixture:
        component_scale = 1.5 * torch.ones(K) + 0.5 * torch.rand(K)
        component_scale.requires_grad_(True)
    else:
        component_scale = torch.ones(K, requires_grad=True)
    if mix_dist == MixtureOfDiagNormals:
        coord_scale = torch.ones(K, D) + 0.5 * torch.rand(K, D)
        coord_scale.requires_grad_(True)
    else:
        coord_scale = torch.ones(D) + 0.5 * torch.rand(D)
        coord_scale.requires_grad_(True)
    if not flat_logits:
        component_logits = (1.5 * torch.rand(K)).requires_grad_(True)
    else:
        component_logits = (0.1 * torch.rand(K)).requires_grad_(True)
    omega = (0.2 * torch.ones(D) + 0.1 * torch.rand(D)).requires_grad_(False)

    _pis = torch.exp(component_logits)
    pis = _pis / _pis.sum()

    if cost_function == 'cosine':
        analytic1 = torch.cos((omega * locs).sum(-1))
        analytic2 = torch.exp(-0.5 * torch.pow(omega * coord_scale * component_scale.unsqueeze(-1), 2.0).sum(-1))
        analytic = (pis * analytic1 * analytic2).sum()
        analytic.backward()
    elif cost_function == 'quadratic':
        analytic = torch.pow(coord_scale * component_scale.unsqueeze(-1), 2.0).sum(-1) + torch.pow(locs, 2.0).sum(-1)
        analytic = (pis * analytic).sum()
        analytic.backward()

    analytic_grads = {}
    analytic_grads['locs'] = locs.grad.clone()
    analytic_grads['coord_scale'] = coord_scale.grad.clone()
    analytic_grads['component_logits'] = component_logits.grad.clone()
    analytic_grads['component_scale'] = component_scale.grad.clone()

    assert locs.grad.shape == locs.shape
    assert coord_scale.grad.shape == coord_scale.shape
    assert component_logits.grad.shape == component_logits.shape
    assert component_scale.grad.shape == component_scale.shape

    coord_scale.grad.zero_()
    component_logits.grad.zero_()
    locs.grad.zero_()
    component_scale.grad.zero_()

    if mix_dist == MixtureOfDiagNormalsSharedCovariance:
        params = {'locs': locs, 'coord_scale': coord_scale, 'component_logits': component_logits}
        if batch_mode:
            locs = locs.unsqueeze(0).expand(n_samples, K, D)
            coord_scale = coord_scale.unsqueeze(0).expand(n_samples, D)
            component_logits = component_logits.unsqueeze(0).expand(n_samples, K)
            dist_params = {'locs': locs, 'coord_scale': coord_scale, 'component_logits': component_logits}
        else:
            dist_params = params
    elif mix_dist == MixtureOfDiagNormals:
        params = {'locs': locs, 'coord_scale': coord_scale, 'component_logits': component_logits}
        if batch_mode:
            locs = locs.unsqueeze(0).expand(n_samples, K, D)
            coord_scale = coord_scale.unsqueeze(0).expand(n_samples, K, D)
            component_logits = component_logits.unsqueeze(0).expand(n_samples, K)
            dist_params = {'locs': locs, 'coord_scale': coord_scale, 'component_logits': component_logits}
        else:
            dist_params = params
    elif mix_dist == GaussianScaleMixture:
        params = {'coord_scale': coord_scale, 'component_logits': component_logits, 'component_scale': component_scale}
        if batch_mode:
            return  # distribution does not support batched parameters
        else:
            dist_params = params

    dist = mix_dist(**dist_params)
    z = dist.rsample(sample_shape=sample_shape)
    assert z.shape == (n_samples, D)
    if cost_function == 'cosine':
        cost = torch.cos((omega * z).sum(-1)).sum() / float(n_samples)
    elif cost_function == 'quadratic':
        cost = torch.pow(z, 2.0).sum() / float(n_samples)
    cost.backward()

    assert_equal(analytic, cost, prec=0.1,
                 msg='bad cost function evaluation for {} test (expected {}, got {})'.format(
                     mix_dist.__name__, analytic.item(), cost.item()))
    logger.debug("analytic_grads_logit: {}"
                 .format(analytic_grads['component_logits'].detach().cpu().numpy()))

    for param_name, param in params.items():
        assert_equal(param.grad, analytic_grads[param_name], prec=0.1,
                     msg='bad {} grad for {} (expected {}, got {})'.format(
                         param_name, mix_dist.__name__, analytic_grads[param_name], param.grad))


@pytest.mark.parametrize('batch_size', [1, 3])
def test_mix_of_diag_normals_shared_cov_log_prob(batch_size):
    locs = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    sigmas = torch.tensor([2.0, 2.0])
    logits = torch.tensor([math.log(0.25), math.log(0.75)])
    value = torch.tensor([0.5, 0.5])
    if batch_size > 1:
        locs = locs.unsqueeze(0).expand(batch_size, 2, 2)
        sigmas = sigmas.unsqueeze(0).expand(batch_size, 2)
        logits = logits.unsqueeze(0).expand(batch_size, 2)
        value = value.unsqueeze(0).expand(batch_size, 2)
    dist = MixtureOfDiagNormalsSharedCovariance(locs, sigmas, logits)
    log_prob = dist.log_prob(value)
    correct_log_prob = 0.25 * math.exp(- 2.25 / 4.0)
    correct_log_prob += 0.75 * math.exp(- 0.25 / 4.0)
    correct_log_prob /= 8.0 * math.pi
    correct_log_prob = math.log(correct_log_prob)
    if batch_size > 1:
        correct_log_prob = [correct_log_prob] * batch_size
    correct_log_prob = torch.tensor(correct_log_prob)
    assert_equal(log_prob, correct_log_prob, msg='bad log prob for MixtureOfDiagNormalsSharedCovariance')


def test_gsm_log_prob():
    sigmas = torch.tensor([2.0, 2.0])
    component_scale = torch.tensor([1.5, 2.5])
    logits = torch.tensor([math.log(0.25), math.log(0.75)])
    dist = GaussianScaleMixture(sigmas, logits, component_scale)
    value = torch.tensor([math.sqrt(0.33), math.sqrt(0.67)])
    log_prob = dist.log_prob(value).item()
    correct_log_prob = 0.25 * math.exp(-0.50 / (4.0 * 2.25)) / 2.25
    correct_log_prob += 0.75 * math.exp(-0.50 / (4.0 * 6.25)) / 6.25
    correct_log_prob /= (2.0 * math.pi) * 4.0
    correct_log_prob = math.log(correct_log_prob)
    assert_equal(log_prob, correct_log_prob, msg='bad log prob for GaussianScaleMixture')


@pytest.mark.parametrize('batch_size', [1, 3])
def test_mix_of_diag_normals_log_prob(batch_size):
    sigmas = torch.tensor([[2.0, 1.5], [1.5, 2.0]])
    locs = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
    logits = torch.tensor([math.log(0.25), math.log(0.75)])
    value = torch.tensor([0.5, 0.25])
    if batch_size > 1:
        locs = locs.unsqueeze(0).expand(batch_size, 2, 2)
        sigmas = sigmas.unsqueeze(0).expand(batch_size, 2, 2)
        logits = logits.unsqueeze(0).expand(batch_size, 2)
        value = value.unsqueeze(0).expand(batch_size, 2)
    dist = MixtureOfDiagNormals(locs, sigmas, logits)
    log_prob = dist.log_prob(value)
    correct_log_prob = 0.25 * math.exp(-0.5 * (0.25 / 4.0 + 0.5625 / 2.25)) / 3.0
    correct_log_prob += 0.75 * math.exp(-0.5 * (2.25 / 2.25 + 0.0625 / 4.0)) / 3.0
    correct_log_prob /= (2.0 * math.pi)
    correct_log_prob = math.log(correct_log_prob)
    if batch_size > 1:
        correct_log_prob = [correct_log_prob] * batch_size
    correct_log_prob = torch.tensor(correct_log_prob)
    assert_equal(log_prob, correct_log_prob, msg='bad log prob for MixtureOfDiagNormals')
