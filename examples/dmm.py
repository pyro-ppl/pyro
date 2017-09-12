import argparse
import torch
import pyro
from torch.autograd import Variable
from pyro.infer.kl_qp import KL_QP
from pyro.distributions import DiagNormal
import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones
from pyro.util import zeros, ones
import cPickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

quick_mode = False
input_dim = 88
z_dim = 100 if not quick_mode else 5
transition_dim = 200 if not quick_mode else 5
emission_dim = 100 if not quick_mode else 5
rnn_dim = 600 if not quick_mode else 5
rnn_num_layers = 2

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

class Combiner(nn.Module):
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        self.lin_z = nn.Linear(z_dim, rnn_dim)
        self.lin_mu = nn.Linear(rnn_dim, z_dim)
        self.lin_sig = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z, h_rnn):
        h_combined = 0.5 * self.tanh(self.lin_z(z) + h_rnn)
        mu = self.lin_mu(h_combined)
        sigma = self.softplus(self.lin_sig(h_combined))
        return mu, sigma

pt_rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                batch_first=True, bidirectional=False, num_layers=rnn_num_layers)
pt_emitter = Emitter(input_dim, z_dim, emission_dim)
pt_trans = GatedTransition(z_dim, transition_dim)
pt_combiner = Combiner(z_dim, rnn_dim)

def model(x_seq):
    emitter = pyro.module("emitter", pt_emitter)
    trans = pyro.module("transition", pt_trans)
    T = x_seq.size(0)
    z_prev = pyro.param("z_0", zeros(z_dim))

    for t in range(1, T + 1):
        z_mu, z_sigma = trans(z_prev)
        z_t = pyro.sample("z_%d" % t, DiagNormal(z_mu, z_sigma))
	emission_probs_t = emitter(z_t)
        pyro.observe("obs_x_%d" % t, dist.bernoulli, x_seq[t - 1, :], emission_probs_t)
        z_prev = z_t

    return z_prev

def reverse(x_seq):
    idx = list(range(x_seq.size(0) - 1, -1, -1))
    idx = Variable(torch.LongTensor(idx))
    return x_seq.index_select(0, idx)

def guide(x_seq):
    h_0 = pyro.param("h_0", zeros(rnn_num_layers, 1, rnn_dim))
    rnn = pyro.module("rnn", pt_rnn)
    combiner = pyro.module("combiner", pt_combiner)
    rnn_output, _ = rnn(reverse(x_seq).view(-1, 1, input_dim), h_0)
    z_prev = pyro.param("z_q_0", zeros(z_dim))
    T = x_seq.size(0)

    for t in range(1, T + 1):
        z_mu, z_sigma = combiner(z_prev, rnn_output[T-t, 0, :])
        z_t = pyro.sample("z_%d" % t, DiagNormal(z_mu, z_sigma))
        z_prev = z_t

    return z_prev

adam_params = {"lr": .0008}
kl_optim = KL_QP(model, guide, pyro.optim(optim.Adam, adam_params))

def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, required=True)
    args = parser.parse_args()

    training_data = cPickle.load(open("jsb_processed.pkl", "rb"))['train']
    training_data_seq_lengths = np.array(training_data['sequence_lengths'], dtype=np.int32)
    training_data_sequences = training_data['array']
    N_train_data = len(training_data_seq_lengths)
    times = [time.time()]

    for epoch in range(args.num_epochs):
        epoch_nll = 0.0
        total_time_slices = 0.0
        for ix in range(N_train_data):
            x_seq = Variable(torch.Tensor(training_data_sequences[0:training_data_seq_lengths[ix], ix, :]))
            loss = kl_optim.step(x_seq)
            total_time_slices += float(training_data_seq_lengths[ix])
            epoch_nll += loss
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        print "[epoch %03d]  %.4f    (dt = %.3f sec)" % (epoch,
            epoch_nll / float(total_time_slices), epoch_time)

if __name__ == '__main__':
    main()
