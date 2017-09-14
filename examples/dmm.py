import argparse
import torch
import pyro
from pyro.infer.kl_qp import KL_QP
from pyro.distributions import DiagNormal
import pyro.distributions as dist
from pyro.util import ng_zeros, ng_ones
from pyro.util import zeros, ones
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import cPickle

quick_mode = False
input_dim = 88
z_dim = 100 if not quick_mode else 5
transition_dim = 200 if not quick_mode else 5
emission_dim = 100 if not quick_mode else 5
rnn_dim = 600 if not quick_mode else 5
rnn_num_layers = 2
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
def model(mini_batch, mini_batch_reversed, mini_batch_mask, T_max):
    emitter = pyro.module("emitter", pt_emitter)
    trans = pyro.module("transition", pt_trans)
    mini_batch_size = mini_batch.size(0)
    z_prev = pyro.param("z_0", zeros(z_dim))

    for t in range(1, T_max + 1):
        z_mu, z_sigma = trans(z_prev)
        z_t = pyro.sample("z_%d" % t, DiagNormal(z_mu, z_sigma),
                          log_pdf_mask=mini_batch_mask[:, t - 1:t].expand(mini_batch_size, z_dim))
        emission_probs_t = emitter(z_t)
        pyro.observe("obs_x_%d" % t, dist.bernoulli, mini_batch[:, t - 1, :], emission_probs_t,
                     log_pdf_mask=mini_batch_mask[:, t - 1:t].expand(mini_batch_size, input_dim))
        z_prev = z_t

    return z_prev


# the guide
def guide(mini_batch, mini_batch_reversed, mini_batch_mask, T_max):
    mini_batch_size = mini_batch.size(0)
    rnn = pyro.module("rnn", pt_rnn)
    combiner = pyro.module("combiner", pt_combiner)
    h_0 = pyro.param("h_0", zeros(rnn_num_layers, 1, rnn_dim))
    rnn_output, _ = rnn(mini_batch_reversed, h_0)
    rnn_output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    z_prev = pyro.param("z_q_0", zeros(z_dim))

    for t in range(1, T_max + 1):
        z_mu, z_sigma = combiner(z_prev, rnn_output[:, T_max - t, :])
        z_t = pyro.sample("z_%d" % t, DiagNormal(z_mu, z_sigma),
                          log_pdf_mask=mini_batch_mask[:, t - 1:t].expand(mini_batch_size, z_dim))
        z_prev = z_t

    return z_prev


# setup and optimization loop
def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, required=True)
    parser.add_argument('-lr', '--learning-rate', type=float, required=True)
    parser.add_argument('-b1', '--beta1', type=float, required=True)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, required=True)
    args = parser.parse_args()

    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, 0.999)}
    kl_optim = KL_QP(model, guide, pyro.optim(optim.Adam, adam_params))

    data = cPickle.load(open("jsb_processed.pkl", "rb"))
    training_data, val_data, test_data = data['train'], data['valid'], data['test']
    training_data_seq_lengths = training_data['sequence_lengths']
    training_data_sequences = training_data['sequences']
    test_data_seq_lengths = test_data['sequence_lengths']
    test_data_sequences = test_data['sequences']
    val_data_seq_lengths = val_data['sequence_lengths']
    val_data_sequences = val_data['sequences']
    N_train_data = len(training_data_seq_lengths)
    N_mini_batches = N_train_data / args.mini_batch_size
    if N_train_data % args.mini_batch_size > 0:
        N_mini_batches += 1

    print "N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" % (N_train_data,\
        np.mean(training_data_seq_lengths), N_mini_batches)
    times = [time.time()]

    def reverse_sequences(mini_batch, seq_lengths):
        for b in range(mini_batch.shape[0]):
            T = seq_lengths[b]
            mini_batch[b, 0:T, :] = mini_batch[b, (T - 1)::-1, :]
        return mini_batch

    def get_mini_batch_mask(mini_batch, seq_lengths):
        mask = np.zeros(mini_batch.shape[0:2])
        for b in range(mini_batch.shape[0]):
            mask[b, 0:seq_lengths[b]] = np.ones(seq_lengths[b])
        return mask

    def get_mini_batch(mini_batch_indices, sequences, seq_lengths):
        seq_lengths = seq_lengths[mini_batch_indices]
        sorted_seq_length_indices = np.argsort(seq_lengths)[::-1]
        sorted_seq_lengths = seq_lengths[sorted_seq_length_indices]
        sorted_mini_batch_indices = mini_batch_indices[sorted_seq_length_indices]
        mini_batch = sequences[sorted_mini_batch_indices, :, :]
        mini_batch_reversed = Variable(torch.Tensor(reverse_sequences(mini_batch, sorted_seq_lengths)))
        mini_batch_reversed = torch.nn.utils.rnn.pack_padded_sequence(mini_batch_reversed,
                                                                      sorted_seq_lengths,
                                                                      batch_first=True)
        mini_batch_time_slices = np.sum(seq_lengths)
        T_max = np.max(seq_lengths)
        mini_batch_mask = Variable(torch.Tensor(get_mini_batch_mask(mini_batch, sorted_seq_lengths)))

        return Variable(torch.Tensor(mini_batch)), mini_batch_reversed, mini_batch_mask,\
            mini_batch_time_slices, T_max

    def do_evaluation(data, seq_lengths):
	N_data, eval_batch_size = data.shape[0], data.shape[0]
	N_eval_batches = N_data / eval_batch_size
	if N_data % eval_batch_size > 0:
	    N_eval_batches += 1
        eval_nll, total_time_slices = 0.0, 0.0

	for which in range(N_eval_batches):
	    mini_batch_start = (which * eval_batch_size)
	    mini_batch_end = np.min([(which + 1) * eval_batch_size, N_data])
	    mini_batch_indices = np.arange(mini_batch_start, mini_batch_end)
	    mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_time_slices, \
		T_max = get_mini_batch(mini_batch_indices, data, seq_lengths)
	    loss = kl_optim.eval_bound(mini_batch, mini_batch_reversed, mini_batch_mask, T_max)
	    total_time_slices += mini_batch_time_slices
	    eval_nll += loss

	return eval_nll / total_time_slices

    for epoch in range(args.num_epochs):
        epoch_nll, total_time_slices = 0.0, 0.0
        shuffled_indices = np.arange(N_train_data)
        np.random.shuffle(shuffled_indices)
        for which in range(N_mini_batches):
            mini_batch_start = (which * args.mini_batch_size)
            mini_batch_end = np.min([(which + 1) * args.mini_batch_size, N_train_data])
            mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
            mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_time_slices, T_max = \
                get_mini_batch(mini_batch_indices, training_data_sequences, training_data_seq_lengths)
            loss = kl_optim.step(mini_batch, mini_batch_reversed, mini_batch_mask, T_max)
            total_time_slices += mini_batch_time_slices
            epoch_nll += loss
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        print "[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" % (epoch,
            epoch_nll / total_time_slices, epoch_time)

        if epoch % val_test_frequency == 0:
            val_nll = do_evaluation(val_data_sequences, val_data_seq_lengths)
            test_nll = do_evaluation(test_data_sequences, test_data_seq_lengths)
            print "[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll)

        if epoch % 100 == 0:
            print "[args]  ", s

if __name__ == '__main__':
    main()
