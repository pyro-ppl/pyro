# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A standard profile HMM model example, using the MuE package.
"""

import argparse
import datetime
import matplotlib.pyplot as plt

import torch
import pyro

from pyro.contrib.mue.models import ProfileHMM

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def main(args):

    torch.manual_seed(0)
    torch.set_default_tensor_type('torch.DoubleTensor')

    small_test = args.test

    if small_test:
        mult_dat = 1
        mult_step = 1
    else:
        mult_dat = 10
        mult_step = 10

    data = torch.cat([torch.tensor([[0., 1.],
                                    [1., 0.],
                                    [0., 1.],
                                    [0., 1.],
                                    [1., 0.],
                                    [0., 0.]])[None, :, :]
                      for j in range(6*mult_dat)] +
                     [torch.tensor([[0., 1.],
                                    [1., 0.],
                                    [1., 0.],
                                    [0., 1.],
                                    [0., 0.],
                                    [0., 0.]])[None, :, :]
                     for j in range(4*mult_dat)], dim=0)
    # Set up inference.
    latent_seq_length, alphabet_length = 6, 2
    adam_params = {"lr": 0.05, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)
    model = ProfileHMM(latent_seq_length, alphabet_length)

    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    n_steps = 10*mult_step

    # Run inference.
    losses = []
    t0 = datetime.datetime.now()
    for step in range(n_steps):
        loss = svi.step(data)
        losses.append(loss)
        if step % 10 == 0:
            print(loss, ' ', datetime.datetime.now() - t0)

    # Plots.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(6, 6))
    plt.plot(losses)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig('phmm_plot.loss_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    precursor_seq = pyro.param("precursor_seq_q_mn").detach()
    precursor_seq_expect = torch.exp(precursor_seq -
                                     precursor_seq.logsumexp(-1, True))
    plt.plot(precursor_seq_expect[:, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of character 1')
    plt.savefig('phmm_plot.precursor_seq_prob_{}.pdf'.format(time_stamp))

    plt.figure(figsize=(6, 6))
    insert = pyro.param("insert_q_mn").detach()
    insert_expect = torch.exp(insert - insert.logsumexp(-1, True))
    plt.plot(insert_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of insert')
    plt.savefig('phmm_plot.insert_prob_{}.pdf'.format(time_stamp))
    plt.figure(figsize=(6, 6))
    delete = pyro.param("delete_q_mn").detach()
    delete_expect = torch.exp(delete - delete.logsumexp(-1, True))
    plt.plot(delete_expect[:, :, 1].numpy())
    plt.xlabel('position')
    plt.ylabel('probability of delete')
    plt.savefig('phmm_plot.delete_prob_{}.pdf'.format(time_stamp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Basic profile HMM model (constant + MuE).")
    parser.add_argument('-t', '--test', action='store_true', default=False,
                        help='small dataset, a few steps')
    args = parser.parse_args()
    main(args)
