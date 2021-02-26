# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A standard profile HMM model.
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
from pyro.contrib.mue.models import ProfileHMM
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

    # Load dataset.
    if args.test:
        dataset = generate_data(args.small)
    else:
        dataset = BiosequenceDataset(args.file, 'fasta', args.alphabet)
    args.batch_size = min([dataset.data_size, args.batch_size])

    # Construct model.
    latent_seq_length = args.latent_seq_length
    if args.latent_seq_length is None:
        latent_seq_length = dataset.max_length
    model = ProfileHMM(latent_seq_length, dataset.alphabet_length,
                       length_model=args.length_model,
                       prior_scale=args.prior_scale,
                       indel_prior_bias=args.indel_prior_bias)

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

    # Plots.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.plots:
        plt.figure(figsize=(6, 6))
        plt.plot(losses)
        plt.xlabel('step')
        plt.ylabel('loss')
        if args.save:
            plt.savefig(os.path.join(
                 args.out_folder,
                 'ProfileHMM_plot.loss_{}.pdf'.format(time_stamp)))

        plt.figure(figsize=(6, 6))
        precursor_seq = pyro.param("precursor_seq_q_mn").detach()
        precursor_seq_expect = torch.exp(precursor_seq -
                                         precursor_seq.logsumexp(-1, True))
        plt.plot(precursor_seq_expect[:, 1].numpy())
        plt.xlabel('position')
        plt.ylabel('probability of character 1')
        if args.save:
            plt.savefig(os.path.join(
                 args.out_folder,
                 'ProfileHMM_plot.precursor_seq_prob_{}.pdf'.format(
                                                                time_stamp)))

        plt.figure(figsize=(6, 6))
        insert = pyro.param("insert_q_mn").detach()
        insert_expect = torch.exp(insert - insert.logsumexp(-1, True))
        plt.plot(insert_expect[:, :, 1].numpy())
        plt.xlabel('position')
        plt.ylabel('probability of insert')
        if args.save:
            plt.savefig(os.path.join(
                 args.out_folder,
                 'ProfileHMM_plot.insert_prob_{}.pdf'.format(time_stamp)))
        plt.figure(figsize=(6, 6))
        delete = pyro.param("delete_q_mn").detach()
        delete_expect = torch.exp(delete - delete.logsumexp(-1, True))
        plt.plot(delete_expect[:, :, 1].numpy())
        plt.xlabel('position')
        plt.ylabel('probability of delete')
        if args.save:
            plt.savefig(os.path.join(
                 args.out_folder,
                 'ProfileHMM_plot.delete_prob_{}.pdf'.format(time_stamp)))

    if args.save:
        pyro.get_param_store().save(os.path.join(
                args.out_folder,
                'ProfileHMM_results.params_{}.out'.format(time_stamp)))
        with open(os.path.join(
                args.out_folder,
                'ProfileHMM_results.input_{}.txt'.format(time_stamp)),
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
    parser.add_argument("-f", "--file", default=None, type=str,
                        help='Input file (fasta format).')
    parser.add_argument("-a", "--alphabet", default='amino-acid',
                        help='Alphabet (amino-acid OR dna).')
    parser.add_argument("-b", "--batch-size", default=10, type=int,
                        help='Batch size.')
    parser.add_argument("-M", "--latent-seq-length", default=None, type=int,
                        help='Latent sequence length.')
    parser.add_argument("-L", "--length-model", default=False, type=bool,
                        help='Model sequence length.')
    parser.add_argument("--prior-scale", default=1., type=float,
                        help='Prior scale parameter (all parameters).')
    parser.add_argument("--indel-prior-bias", default=10., type=float,
                        help='Indel prior bias parameter.')
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