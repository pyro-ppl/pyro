# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np
import torch
from torch.optim import Adam

import pyro

from pyro.contrib.mue.dataloaders import BiosequenceDataset
from pyro.contrib.mue.models import ProfileHMM, FactorMuE

from pyro.optim import MultiStepLR


@pytest.mark.parametrize('indel_factor_dependence', [False, True])
@pytest.mark.parametrize('z_prior_distribution', ['Normal', 'Laplace'])
@pytest.mark.parametrize('ARD_prior', [False, True])
@pytest.mark.parametrize('substitution_matrix', [False, True])
@pytest.mark.parametrize('length_model', [False, True])
def test_FactorMuE_smoke(indel_factor_dependence, z_prior_distribution,
                         ARD_prior, substitution_matrix, length_model):
    # Setup dataset.
    seqs = ['BABBA', 'BAAB', 'BABBB']
    alph = ['A', 'B']
    dataset = BiosequenceDataset(seqs, 'list', alph)

    # Infer.
    z_dim = 2
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': 0.1},
                             'milestones': [20, 100, 1000, 2000],
                             'gamma': 0.5})
    model = FactorMuE(dataset.max_length, dataset.alphabet_length, z_dim,
                      indel_factor_dependence=indel_factor_dependence,
                      z_prior_distribution=z_prior_distribution,
                      ARD_prior=ARD_prior,
                      substitution_matrix=substitution_matrix,
                      length_model=length_model)
    n_epochs = 5
    batch_size = 2
    losses = model.fit_svi(dataset, n_epochs, batch_size, scheduler)

    # Reconstruct.
    recon = model.reconstruct_precursor_seq(dataset, 1, pyro.param)

    assert not np.isnan(losses[-1])
    assert recon.shape == (1, max([len(seq) for seq in seqs]), len(alph))
