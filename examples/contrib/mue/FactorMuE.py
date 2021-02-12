# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A PCA model with a MuE emission (FactorMuE). Uses the MuE package.
"""

import argparse
import datetime
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
import pyro

from pyro.contrib.mue.dataloaders import BiosequenceDataset
from pyro.contrib.mue.models import FactorMuE

from pyro.optim import MultiStepLR


def main(args):

    torch.manual_seed(9)
    torch.set_default_tensor_type('torch.DoubleTensor')

    small_test = args.test

    if small_test:
        mult_dat = 1
        mult_step = 1
    else:
        mult_dat = 10
        mult_step = 400

    # Construct example dataset.
    seqs = ['BABBA']*mult_dat + ['BAAB']*mult_dat + ['BABBB']*mult_dat
    dataset = BiosequenceDataset(seqs, 'list', ['A', 'B'])

    # Set up inference.
    z_dim = 2
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': 0.1},
                             'milestones': [20, 100, 1000, 2000],
                             'gamma': 0.5})
    model = FactorMuE(dataset.max_length, dataset.alphabet_length, z_dim,
                      substitution_matrix=False)
    n_epochs = 10*mult_step
    batch_size = len(dataset)

    # Infer.
    losses = model.fit_svi(dataset, n_epochs, batch_size, scheduler)

    # Plots.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(6, 6))
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('FactorMuE_plot.loss_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    latent = model.encoder(dataset.seq_data)[0].detach()
    plt.scatter(latent[:, 0], latent[:, 1])
    plt.xlabel('z_1')
    plt.ylabel('z_2')
    plt.savefig('FactorMuE_plot.latent_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    insert = pyro.param("insert_q_mn").detach()
    insert_expect = torch.exp(insert - insert.logsumexp(-1, True))
    plt.plot(insert_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of insert')
    plt.savefig('FactorMuE_plot.insert_prob_{}.pdf'.format(time_stamp))
    plt.figure(figsize=(6, 6))
    delete = pyro.param("delete_q_mn").detach()
    delete_expect = torch.exp(delete - delete.logsumexp(-1, True))
    plt.plot(delete_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of delete')
    plt.savefig('FactorMuE_plot.delete_prob_{}.pdf'.format(time_stamp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Basic Factor MuE model.")
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='small dataset, a few steps')
    args = parser.parse_args()
    main(args)
