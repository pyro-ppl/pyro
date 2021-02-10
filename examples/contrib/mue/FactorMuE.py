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

from pyro.contrib.mue.models import FactorMuE

from pyro.infer import SVI, Trace_ELBO
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
    xs = [torch.tensor([[0., 1.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [1., 0.],
                        [0., 0.]]),
          torch.tensor([[0., 1.],
                        [1., 0.],
                        [1., 0.],
                        [0., 1.],
                        [0., 0.],
                        [0., 0.]]),
          torch.tensor([[0., 1.],
                        [1., 0.],
                        [0., 1.],
                        [0., 1.],
                        [0., 1.],
                        [0., 0.]])]
    data = torch.cat([xs[0][None, :, :] for j in range(6*mult_dat)] +
                     [xs[1][None, :, :] for j in range(4*mult_dat)] +
                     [xs[2][None, :, :] for j in range(4*mult_dat)], dim=0)
    # Set up inference.
    obs_seq_length, alphabet_length, z_dim = 6, 2, 2
    # adam_params = {"lr": 0.1, "betas": (0.90, 0.999)}
    scheduler = MultiStepLR({'optimizer': Adam,
                             'optim_args': {'lr': 0.1},
                             'milestones': [20, 100, 1000, 2000],
                             'gamma': 0.5})
    # optimizer = Adam(adam_params)
    model = FactorMuE(obs_seq_length, alphabet_length, z_dim,
                      substitution_matrix=False)

    svi = SVI(model.model, model.guide, scheduler, loss=Trace_ELBO())
    n_steps = 10*mult_step

    # Run inference.
    losses = []
    t0 = datetime.datetime.now()
    for step in range(n_steps):

        loss = svi.step(data)
        losses.append(loss)
        scheduler.step()
        if step % 10 == 0:
            print(step, loss, ' ', datetime.datetime.now() - t0)

    # Plots.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(6, 6))
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('FactorMuE_plot.loss_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    latent = model.encoder(data)[0].detach()
    plt.scatter(latent[:, 0], latent[:, 1])
    plt.xlabel('z_1')
    plt.ylabel('z_2')
    plt.savefig('FactorMuE_plot.latent_{}.pdf'.format(time_stamp))

    # plt.figure(figsize=(6, 6))
    # decoder_bias = pyro.param('decoder$$$f.bias').detach()
    # decoder_bias = decoder_bias.reshape(
    #                 [-1, 2, model.latent_seq_length+1, model.alphabet_length])
    # plt.plot(decoder_bias[0, 0, :, 1])
    # plt.xlabel('position')
    # plt.ylabel('bias for character 1')
    # plt.savefig('FactorMuE_plot.decoder_bias_{}.pdf'.format(time_stamp))

    for xi, x in enumerate(xs):
        reconstruct_x = model.reconstruct_precursor_seq(x, pyro.param)
        plt.figure(figsize=(6, 6))
        plt.plot(reconstruct_x[0, :, 1], label="reconstruct")
        plt.plot(x[:, 1], label="data")
        plt.xlabel('position')
        plt.ylabel('probability of character 1')
        plt.legend()
        plt.savefig('FactorMuE_plot.reconstruction_{}_{}.pdf'.format(
                        xi, time_stamp))

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
