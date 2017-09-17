import argparse
import torch
import pyro
from pyro.infer.kl_qp import KL_QP
import pyro.distributions as dist
from pyro.util import ng_ones, zeros
import torch.nn as nn
# from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import cPickle
import polyphonic_data_loader as poly
from pyro.optim import ClippedAdam
import logging


input_dim = 88
z_dim = 100
transition_dim = 200
emission_dim = 100
rnn_dim = 600
rnn_num_layers = 1
val_test_frequency = 10


# parameterizes p(x_t | z_t)
class Emitter(nn.Module):
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


# parameterizes p(z_t | z_{t-1})
class GatedTransition(nn.Module):
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
        mu = (ng_ones(g.size()) - g) * self.lin_mu(z) + g * h
        sigma = self.softplus(self.lin_sig(self.relu(h)))
        return mu, sigma


# parameterizes q(z_t | z_{t-1}, x_{t:T})
# the dependence on x_{t:T} is through the hidden state of the RNN (see pt_rnn below)
class Combiner(nn.Module):
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


# instantiate pytorch modules that make up the model and the inference network
pt_emitter = Emitter(input_dim, z_dim, emission_dim)
pt_trans = GatedTransition(z_dim, transition_dim)
pt_combiner = Combiner(z_dim, rnn_dim)
pt_rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                batch_first=True, bidirectional=False, num_layers=rnn_num_layers)


# the model
def model(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths,
          annealing_factor=1.0):

    T_max = np.max(mini_batch_seq_lengths)
    emitter = pyro.module("emitter", pt_emitter)
    trans = pyro.module("transition", pt_trans)
    z_prev = pyro.param("z_0", zeros(z_dim))

    for t in range(1, T_max + 1):
        z_mu, z_sigma = trans(z_prev)
        z_t = pyro.sample("z_%d" % t, dist.DiagNormal(z_mu, z_sigma),
                          log_pdf_mask=annealing_factor * mini_batch_mask[:, t - 1:t])
        emission_probs_t = emitter(z_t)
        pyro.observe("obs_x_%d" % t, dist.bernoulli, mini_batch[:, t - 1, :], emission_probs_t,
                     log_pdf_mask=mini_batch_mask[:, t - 1:t])
        z_prev = z_t

    return z_prev


# the guide
def guide(mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths,
          annealing_factor=1.0):

    T_max = np.max(mini_batch_seq_lengths)
    rnn = pyro.module("rnn", pt_rnn)
    combiner = pyro.module("combiner", pt_combiner)
    h_0 = pyro.param("h_0", zeros(rnn_num_layers, 1, rnn_dim))
    rnn_output, _ = rnn(mini_batch_reversed, h_0)
    rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
    z_prev = pyro.param("z_q_0", zeros(z_dim))

    for t in range(1, T_max + 1):
        z_mu, z_sigma = combiner(z_prev, rnn_output[:, t - 1, :])
        z_t = pyro.sample("z_%d" % t, dist.DiagNormal(z_mu, z_sigma),
                          log_pdf_mask=annealing_factor * mini_batch_mask[:, t - 1:t])
        z_prev = z_t

    return z_prev


# setup, training, and evaluation
def main():
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
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=args.log, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
	logging.info(s)
    log(args)

    # ingest data from disk
    data = cPickle.load(open("jsb_processed.pkl", "rb"))
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']
    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = np.sum(training_seq_lengths)
    N_mini_batches = N_train_data / args.mini_batch_size +\
        int(N_train_data % args.mini_batch_size > 0)

    # package val/test data for model/guide
    def rep(x):
        y = np.repeat(x, 5, axis=0)
        return y

    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    val_batch, val_batch_reversed, val_batch_mask, _ = poly.get_mini_batch(\
	np.arange(5*val_data_sequences.shape[0]), rep(val_data_sequences), val_seq_lengths)
    test_batch, test_batch_reversed, test_batch_mask, _ = poly.get_mini_batch(\
	np.arange(5*test_data_sequences.shape[0]), rep(test_data_sequences), test_seq_lengths)

    log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
        (N_train_data, np.mean(training_seq_lengths), N_mini_batches))
    times = [time.time()]

    # setup optimizers
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    elbo_train = KL_QP(model, guide, pyro.optim(ClippedAdam, adam_params))
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
                                      training_seq_lengths)
            loss = elbo_train.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                                   mini_batch_seq_lengths, annealing_factor)
            if epoch < 1:
                log("minibatch loss:  %.4f  [annealing factor: %.4f]" % (loss, annealing_factor))
            epoch_nll += loss
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
              (epoch, epoch_nll / N_train_time_slices, epoch_time))

        if epoch % val_test_frequency == 0:
            val_nll = elbo_train.eval_objective(val_batch, val_batch_reversed, val_batch_mask,
                                                val_seq_lengths) / np.sum(val_seq_lengths)
            test_nll = elbo_train.eval_objective(test_batch, test_batch_reversed, test_batch_mask,
                                                 test_seq_lengths) / np.sum(test_seq_lengths)
            log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))

if __name__ == '__main__':
    main()
