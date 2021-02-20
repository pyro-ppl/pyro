# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A PCA model with a MuE emission (FactorMuE).
"""

import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

import pyro
from pyro.contrib.mue.dataloaders import BiosequenceDataset
from pyro.contrib.mue.models import FactorMuE
from pyro.optim import MultiStepLR


def generate_data(small_test):
    """Generate example dataset."""
    if small_test:
        mult_dat = 1
    else:
        mult_dat = 10

    seqs = ['BABBA']*mult_dat + ['BAAB']*mult_dat + ['BABBB']*mult_dat
    dataset = BiosequenceDataset(seqs, 'list', ['A', 'B'])

    return dataset


def main(args):

    pyro.set_rng_seed(args.rng_seed)

    # Construct example dataset.
    if args.test:
        dataset = generate_data(args.small)
    else:
        dataset = BiosequenceDataset(args.file, 'fasta', args.alphabet)
    args.batch_size = min([dataset.data_size, args.batch_size])

    # Construct model.
    model = FactorMuE(dataset.max_length, dataset.alphabet_length, args.z_dim,
                      batch_size=args.batch_size,
                      latent_seq_length=args.latent_seq_length,
                      indel_factor_dependence=args.indel_factor,
                      indel_prior_scale=args.indel_prior_scale,
                      indel_prior_bias=args.indel_prior_bias,
                      inverse_temp_prior=args.inverse_temp_prior,
                      weights_prior_scale=args.weights_prior_scale,
                      offset_prior_scale=args.offset_prior_scale,
                      z_prior_distribution=args.z_prior,
                      ARD_prior=args.ARD_prior,
                      substitution_matrix=args.substitution_matrix,
                      substitution_prior_scale=args.substitution_prior_scale,
                      latent_alphabet_length=args.latent_alphabet,
                      length_model=args.length_model)

    # Infer.
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': args.learning_rate},
                             'milestones': json.loads(args.milestones),
                             'gamma': args.learning_gamma})
    if args.test and not args.small:
        n_epochs = 100
    else:
        n_epochs = args.n_epochs
    losses = model.fit_svi(dataset, n_epochs, args.batch_size, scheduler)

    # Plot and save.
    if args.plots:
        time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.figure(figsize=(6, 6))
        plt.plot(losses)
        plt.xlabel('step')
        plt.ylabel('loss')
        if args.save:
            plt.savefig(os.path.join(
                 args.out_folder,
                 'FactorMuE_plot.loss_{}.pdf'.format(time_stamp)))

        plt.figure(figsize=(6, 6))
        latent = model.encoder(dataset.seq_data)[0].detach()
        plt.scatter(latent[:, 0], latent[:, 1])
        plt.xlabel('z_1')
        plt.ylabel('z_2')
        if args.save:
            plt.savefig(os.path.join(
                 args.out_folder,
                 'FactorMuE_plot.latent_{}.pdf'.format(time_stamp)))

        if not args.indel_factor:
            plt.figure(figsize=(6, 6))
            insert = pyro.param("insert_q_mn").detach()
            insert_expect = torch.exp(insert - insert.logsumexp(-1, True))
            plt.plot(insert_expect[:, :, 1].numpy())
            plt.xlabel('position')
            plt.ylabel('probability of insert')
            if args.save:
                plt.savefig(os.path.join(
                     args.out_folder,
                     'FactorMuE_plot.insert_prob_{}.pdf'.format(time_stamp)))
            plt.figure(figsize=(6, 6))
            delete = pyro.param("delete_q_mn").detach()
            delete_expect = torch.exp(delete - delete.logsumexp(-1, True))
            plt.plot(delete_expect[:, :, 1].numpy())
            plt.xlabel('position')
            plt.ylabel('probability of delete')
            if args.save:
                plt.savefig(os.path.join(
                     args.out_folder,
                     'FactorMuE_plot.delete_prob_{}.pdf'.format(time_stamp)))
    if args.save:
        pyro.get_param_store().save(os.path.join(
                args.out_folder,
                'FactorMuE_results.params_{}.out'.format(time_stamp)))
        with open(os.path.join(
                args.out_folder,
                'FactorMuE_results.input_{}.txt'.format(time_stamp)),
                'w') as ow:
            ow.write('[args]\n')
            for elem in list(args.__dict__.keys()):
                ow.write('{} = {}\n'.format(elem, args.__getattribute__(elem)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Factor MuE model.")
    parser.add_argument("--test", action='store_true', default=False,
                        help='Run with generated example dataset.')
    parser.add_argument("--small", action='store_true', default=False,
                        help='Run with small example dataset.')
    parser.add_argument("-r", "--rng-seed", default=0, type=int)
    parser.add_argument("-f", "--file", default=None,
                        help='Input file (fasta format).')
    parser.add_argument("-a", "--alphabet", default='amino-acid',
                        help='Alphabet (amino-acid OR dna).')
    parser.add_argument("-zdim", "--z-dim", default=2, type=int,
                        help='z space dimension.')
    parser.add_argument("-b", "--batch-size", default=10, type=int,
                        help='Batch size.')
    parser.add_argument("-M", "--latent-seq-length", default=None,
                        help='Latent sequence length.')
    parser.add_argument("-idfac", "--indel-factor", default=False, type=bool,
                        help='Indel parameters depend on latent variable.')
    parser.add_argument("-zdist", "--z-prior", default='Normal',
                        help='Latent prior distribution (normal or Laplace).')
    parser.add_argument("-ard", "--ARD-prior", default=False, type=bool,
                        help='Use automatic relevance detection prior.')
    parser.add_argument("-sub", "--substitution-matrix", default=True, type=bool,
                        help='Use substitution matrix.')
    parser.add_argument("-D", "--latent-alphabet", default=None,
                        help='Latent alphabet length.')
    parser.add_argument("-L", "--length-model", default=False, type=bool,
                        help='Model sequence length.')
    parser.add_argument("--indel-prior-scale", default=1., type=float,
                        help=('Indel prior scale parameter ' +
                              '(when indel-factor=False).'))
    parser.add_argument("--indel-prior-bias", default=10., type=float,
                        help='Indel prior bias parameter.')
    parser.add_argument("--inverse-temp-prior", default=100., type=float,
                        help='Inverse temperature prior mean.')
    parser.add_argument("--weights-prior-scale", default=1., type=float,
                        help='Factor parameter prior scale.')
    parser.add_argument("--offset-prior-scale", default=1., type=float,
                        help='Offset parameter prior scale.')
    parser.add_argument("--substitution-prior-scale", default=10., type=float,
                        help='Substitution matrix prior scale.')
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float,
                        help='Learning rate for Adam optimizer.')
    parser.add_argument("--milestones", default='[]', type=str,
                        help='Milestones for multistage learning rate.')
    parser.add_argument("--learning-gamma", default=0.5, type=float,
                        help='Gamma parameter for multistage learning rate.')
    parser.add_argument("-e", "--n-epochs", default=10, type=int,
                        help='Number of epochs of training.')
    parser.add_argument("-p", "--plots", default=True, type=bool,
                        help='Make plots.')
    parser.add_argument("-s", "--save", default=True, type=bool,
                        help='Save plots and results.')
    parser.add_argument("-outf", "--out-folder", default='.',
                        help='Folder to save plots.')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    main(args)
