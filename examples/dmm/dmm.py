"""
An implementation of a Deep Markov Model in pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae. We
also explore including normalizing flows in the posterior over the latents (in which case
analytic formulae for the KL divergences are in any case unavailable).

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import torch
import pyro
import numpy as np
import time
from pyro.infer.kl_qp import KL_QP
import pyro.distributions as dist
from pyro.util import ng_ones, zeros
import torch.nn as nn
from pyro.distributions.transformed_distribution import InverseAutoregressiveFlow
from pyro.distributions.transformed_distribution import TransformedDistribution
import six.moves.cPickle as pickle
import polyphonic_data_loader as poly
from pyro.optim import ClippedAdam
from os.path import join, dirname
import logging


class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood p(x_t | z_t)
    """
    def __init__(self, input_dim, z_dim, emission_dim):
        super(Emitter, self).__init__()
        self.lin1 = nn.Linear(z_dim, emission_dim)
        self.lin2 = nn.Linear(emission_dim, emission_dim)
        self.lin3 = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h1 = self.relu(self.lin1(z))
        h2 = self.relu(self.lin2(h1))
        ps = self.sigmoid(self.lin3(h2))
        return ps


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    """
    def __init__(self, z_dim, transition_dim):
        super(GatedTransition, self).__init__()
        self.lin_g1 = nn.Linear(z_dim, transition_dim)
        self.lin_g2 = nn.Linear(transition_dim, z_dim)
        self.lin_h1 = nn.Linear(z_dim, transition_dim)
        self.lin_h2 = nn.Linear(transition_dim, z_dim)
        self.lin_mu = nn.Linear(z_dim, z_dim)
        self.lin_mu.weight.data = torch.eye(z_dim)
        self.lin_mu.bias.data = torch.zeros(z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, z):
        g = self.sigmoid(self.lin_g2(self.relu(self.lin_g1(z))))
        h = self.lin_h2(self.relu(self.lin_h1(z)))
        mu = (ng_ones(g.size()).type_as(g) - g) * self.lin_mu(z) + g * h
        sigma = self.softplus(self.lin_sig(self.relu(h)))
        return mu, sigma


class Combiner(nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block of the
    guide (i.e. the variational distribution). The dependence on x_{t:T} is through the
    hidden state of the RNN (see pt_rnn below)
    """
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        self.lin_z = nn.Linear(z_dim, rnn_dim)
        self.lin_mu = nn.Linear(rnn_dim, z_dim)
        self.lin_sig = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z, h_rnn):
        h_combined = 0.5 * (self.tanh(self.lin_z(z)) + h_rnn)
        mu = self.lin_mu(h_combined)
        sigma = self.softplus(self.lin_sig(h_combined))
        return mu, sigma


class DMM(nn.Module):
    """
    This pytorch Module encapsulates the model as well as the variational distribution (the guide)
    for the Deep Markov Model
    """
    def __init__(self, input_dim, z_dim, emission_dim, transition_dim, rnn_dim, args):
        super(DMM, self).__init__()
        # instantiate pytorch modules
        self.pt_emitter = Emitter(input_dim, z_dim, emission_dim)
        self.pt_trans = GatedTransition(z_dim, transition_dim)
        self.pt_combiner = Combiner(z_dim, rnn_dim)
        self.pt_rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                             batch_first=True, bidirectional=False, num_layers=args.rnn_num_layers,
                             dropout=args.rnn_dropout_rate)

        self.args = args
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.emission_dim = emission_dim
        self.transition_dim = transition_dim
        self.rnn_dim = rnn_dim
        self.pt_iafs = [InverseAutoregressiveFlow(z_dim, 100) for _ in range(args.num_iafs)]
        if cuda:
            for pt_iaf in self.pt_iafs:
                pt_iaf.cuda()

    # the model
    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        T_max = np.max(mini_batch_seq_lengths)
        emitter = pyro.module("emitter", self.pt_emitter)
        trans = pyro.module("transition", self.pt_trans)
        z_prev = pyro.param("z_0", zeros(self.z_dim, type_as=mini_batch.data))

        for t in range(1, T_max + 1):
            z_mu, z_sigma = trans(z_prev)
            z_t = pyro.sample("z_%d" % t, dist.DiagNormal(z_mu, z_sigma),
                              log_pdf_mask=annealing_factor * mini_batch_mask[:, t - 1:t])
            emission_probs_t = emitter(z_t)
            pyro.observe("obs_x_%d" % t, dist.bernoulli, mini_batch[:, t - 1, :], emission_probs_t,
                         log_pdf_mask=mini_batch_mask[:, t - 1:t])
            z_prev = z_t

        return z_prev

    # the guide (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        T_max = np.max(mini_batch_seq_lengths)
        rnn = pyro.module("rnn", pt_rnn)
        combiner = pyro.module("combiner", self.pt_combiner)

        # complicated rnn behavior for gpus
        # must provide the whole state tensor at time of gpu running
        # on gpu, must provide batch_size number of hidden states
        if 'cuda' in mini_batch.data.type():
            h_0 = pyro.param("h_0", zeros(pt_rnn.num_layers, 1, self.rnn_dim, type_as=mini_batch.data))
            h_0 = h_0.expand(*[h_0.data.shape[0], mini_batch.data.shape[0], h_0.data.shape[2]]).contiguous()
        else:
            # on cpu, hidden size starts as 1 and is created
            h_0 = pyro.param("h_0", zeros(pt_rnn.num_layers, 1, self.rnn_dim, type_as=mini_batch.data))

        rnn_output, _ = rnn(mini_batch_reversed, h_0)
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        z_prev = pyro.param("z_q_0", zeros(self.z_dim, type_as=mini_batch.data))

        for t in range(1, T_max + 1):
            z_mu, z_sigma = combiner(z_prev, rnn_output[:, t - 1, :])
            z_dist = dist.DiagNormal(z_mu, z_sigma)
            iafs = [pyro.module("iaf_%d" % i, pt_iaf) for i, pt_iaf in enumerate(self.pt_iafs)]
            if self.num_iafs > 0:
                z_dist = TransformedDistribution(z_dist, iafs)
            z_t = pyro.sample("z_%d" % t, z_dist,
                              log_pdf_mask=annealing_factor * mini_batch_mask[:, t - 1:t])
            z_prev = z_t

        return z_prev


# setup, training, and evaluation
def main(num_epochs=2000, learning_rate=0.0008, beta1=.9, beta2=.999,
         clip_norm=25., lr_decay=1.0, weight_decay=0.0,
         mini_batch_size=20, annealing_epochs=500,
         minimum_annealing_factor=0.0, rnn_dropout_rate=0.0,
         rnn_num_layers=1, log='dmm.log', cuda=False, num_iafs=0):

    # ensure ints
    # TODO: Remove this, temp hyper param logic
    num_epochs = int(num_epochs)
    mini_batch_size = int(mini_batch_size)
    annealing_epochs = int(annealing_epochs)
    rnn_num_layers = int(rnn_num_layers)

    input_dim = 88
    z_dim = 100
    transition_dim = 200
    emission_dim = 100
    rnn_dim = 600
    val_test_frequency = 20
    n_eval_samples_inner = 5
    n_eval_samples_outer = 100

    # setup logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=args.log, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    log(args)

    pt_rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                    batch_first=True, bidirectional=False, num_layers=args.rnn_num_layers,
                    dropout=args.rnn_dropout_rate)

    jsb_file_loc = join(dirname(__file__), "jsb_processed.pkl")
    # ingest data from disk
    data = pickle.load(open(jsb_file_loc, "rb"))
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']
    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = np.sum(training_seq_lengths)
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                         int(N_train_data % args.mini_batch_size > 0))

    # package val/test data for model/guide
    def rep(x):
        y = np.repeat(x, n_eval_samples_inner, axis=0)
        return y

    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = poly.get_mini_batch(
        np.arange(n_eval_samples_inner * val_data_sequences.shape[0]), rep(val_data_sequences),
        val_seq_lengths, volatile=True, cuda=cuda)
    test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = poly.get_mini_batch(
        np.arange(n_eval_samples_inner * test_data_sequences.shape[0]), rep(test_data_sequences),
        test_seq_lengths, volatile=True, cuda=cuda)

    log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
        (N_train_data, np.mean(training_seq_lengths), N_mini_batches))
    times = [time.time()]

    # create the dmm
    dmm = DMM(input_dim, z_dim, emission_dim, transition_dim, rnn_dim, num_iafs, cuda)

    # easy fix, turn dmm into cuda dmm
    if cuda:
        dmm = dmm.cuda()
        pt_rnn = pt_rnn.cuda()

    # setup optimizers
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    elbo_train = KL_QP(dmm.model, dmm.guide, pyro.optim(ClippedAdam, adam_params))
    annealing_factor = 1.0

    for epoch in range(args.num_epochs):
        epoch_nll = 0.0
        shuffled_indices = np.arange(N_train_data)
        np.random.shuffle(shuffled_indices)
        for which in range(N_mini_batches):
            if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
                min_af = args.minimum_annealing_factor
                annealing_factor = min_af + (1.0 - min_af) * \
                    (float(which + epoch * N_mini_batches + 1) /
                     float(args.annealing_epochs * N_mini_batches))
            mini_batch_start = (which * args.mini_batch_size)
            mini_batch_end = np.min([(which + 1) * args.mini_batch_size, N_train_data])
            mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
            mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
                = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                      training_seq_lengths, cuda=cuda)
            loss = elbo_train.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                                   mini_batch_seq_lengths, pt_rnn, annealing_factor)
            if epoch < 1:
                log("minibatch loss:  %.4f  [annealing factor: %.4f]" % (loss, annealing_factor))
            epoch_nll += loss
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
            (epoch, epoch_nll / N_train_time_slices, epoch_time))

        if epoch % val_test_frequency == 0 and False:
            pt_rnn.eval()
            val_nlls, test_nlls = [], []
            for _ in range(n_eval_samples_outer):
                val_nll = elbo_train.eval_objective(val_batch, val_batch_reversed, val_batch_mask,
                                                    val_seq_lengths, pt_rnn) / np.sum(val_seq_lengths)
                test_nll = elbo_train.eval_objective(test_batch, test_batch_reversed, test_batch_mask,
                                                     test_seq_lengths, pt_rnn) / np.sum(test_seq_lengths)
                val_nlls.append(val_nll)
                test_nlls.append(test_nll)
            val_nll = np.mean(val_nlls)
            test_nll = np.mean(test_nlls)
            pt_rnn.train()
            log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))

    # when finished, do eval and send back validation
    # for hyper param search
    pt_rnn.eval()
    val_nlls, test_nlls = [], []
    for _ in range(n_eval_samples_outer):
        val_nll = elbo_train.eval_objective(val_batch, val_batch_reversed, val_batch_mask,
                                            val_seq_lengths, pt_rnn) / np.sum(val_seq_lengths)
        val_nlls.append(val_nll)
        test_nll = elbo_train.eval_objective(test_batch, test_batch_reversed, test_batch_mask,
                                             test_seq_lengths, pt_rnn) / np.sum(test_seq_lengths)
        test_nlls.append(test_nll)
    test_nll, val_nll = np.mean(test_nlls), np.mean(val_nlls)
    log("[validation score final epoch]  %.5f" % val_nll)
    log("[test score final epoch]  %.5f" % test_nll)
    return (test_nll, val_nll)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=2000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0008)
    parser.add_argument('-b1', '--beta1', type=float, default=0.90)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=25.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=1.0)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=500)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.0)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.0)
    parser.add_argument('-rnl', '--rnn-num-layers', type=int, default=1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(**vars(args))
