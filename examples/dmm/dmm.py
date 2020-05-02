# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae.
We also illustrate the use of normalizing flows in the variational distribution (in which
case analytic formulae for the KL divergences are in any case unavailable).

Reference:

[1] Structured Inference Networks for Nonlinear State Space Models [arXiv:1609.09869]
    Rahul G. Krishnan, Uri Shalit, David Sontag
"""

import argparse
import time
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
from torch.distributions import constraints

import polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import affine_autoregressive
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, config_enumerate
from pyro.optim import ClippedAdam
from util import get_logger

from pyro.distributions.torch_distribution import TorchDistribution

from pyro.ops.gaussian import gaussian_tensordot, matrix_and_mvn_to_gaussian, mvn_to_gaussian
from pyro.distributions.hmm import _sequential_gaussian_filter_sample, _sequential_gaussian_tensordot


class Emitter(nn.Module):
    """
    Parameterizes the bernoulli observation likelihood `p(x_t | z_t)`
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        # initialize the three linear transformations used in the neural network
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        # initialize the two non-linearities used in the neural network
        self.relu = nn.ReLU()

    def forward(self, z_t):
        """
        Given the latent z at a particular time step t we return the vector of
        probabilities `ps` that parameterizes the bernoulli distribution `p(x_t|z_t)`
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        ps = torch.sigmoid(self.lin_hidden_to_input(h2))
        return ps


class GatedTransition(nn.Module):
    """
    Parameterizes the gaussian latent transition probability `p(z_t | z_{t-1})`
    See section 5 in the reference for comparison.
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        # initialize the six linear transformations used in the neural network
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)
        # modify the default initialization of lin_z_to_loc
        # so that it's starts out as the identity function
        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)
        # initialize the three non-linearities used in the neural network
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        """
        Given the latent `z_{t-1}` corresponding to the time step t-1
        we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution `p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    Parameterizes `q(z_t | z_{t-1}, x_{t:T})`, which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on `x_{t:T}` is
    through the hidden state of the RNN (see the PyTorch module `rnn` below)
    """

    def __init__(self, z_dim, rnn_dim, guide_family):
        super().__init__()
        self.z_dim = z_dim
        self.guide_family = guide_family
        # initialize the three linear transformations used in the neural network
        if self.guide_family == 'markov':
            self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim // 4)
            self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim // 4)
            self.lin_hidden_to_mat = nn.Linear(rnn_dim, z_dim * z_dim // 4)
        elif self.guide_family == 'meanfield':
            self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
            self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        # initialize the two non-linearities used in the neural network
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden
        state of the RNN `h(x_{t:T})` we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution `q(z_t | z_{t-1}, x_{t:T})`
        """
        loc = self.lin_hidden_to_loc(h_rnn)
        scale = self.softplus(self.lin_hidden_to_scale(h_rnn))
        if self.guide_family == 'markov':
            mat = self.lin_hidden_to_mat(h_rnn)
            mat = mat.reshape(mat.size(0), mat.size(1), self.z_dim, self.z_dim // 4)
            return loc, scale, mat
        else:
            return loc, scale, None


class DMM(nn.Module):
    """
    This PyTorch Module encapsulates the model as well as the
    variational distribution (the guide) for the Deep Markov Model
    """

    def __init__(self, input_dim=88, z_dim=64, emission_dim=100,
                 transition_dim=200, rnn_dim=300, num_layers=1, rnn_dropout_rate=0.0,
                 use_cuda=False, guide_family="meanfield"):
        super().__init__()
        self.guide_family = guide_family
        assert guide_family in ['meanfield', 'markov']
        self.z_dim = z_dim
        # instantiate PyTorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim, guide_family)
        # dropout just takes effect on inner layers of rnn
        rnn_dropout_rate = 0. if num_layers == 1 else rnn_dropout_rate
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=num_layers,
                          dropout=rnn_dropout_rate)

        # define a (trainable) parameters z_0 and z_q_0 that help define the probability
        # distributions p(z_1) and q(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        # define a (trainable) parameter for the initial hidden state of the rnn
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.use_cuda = use_cuda
        # if on gpu cuda-ize all PyTorch (sub)modules
        if use_cuda:
            self.cuda()

    # the model p(x_{1:T} | z_{1:T}) p(z_{1:T})
    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        T_max = mini_batch.size(1)

        pyro.module("dmm", self)

        ones = torch.ones(mini_batch.size(0), T_max, self.z_0.size(0), dtype=self.z_0.dtype, device=self.z_0.device)
        # note the mask
        z = pyro.sample("z", dist.Normal(0.0, ones).mask(False).to_event(3))

        z_0 = self.z_0.expand(mini_batch.size(0), 1, self.z_0.size(0))
        # shift z for autoregressive conditioning
        z_shift = torch.cat([z_0, z[:, :-1, :]], dim=-2)
        z_loc, z_scale = self.trans(z_shift)

        mask = mini_batch_mask.unsqueeze(-1)

        with poutine.scale(scale=annealing_factor):
            # actually compute p(z)
            pyro.sample("z_aux", dist.Normal(z_loc, z_scale).mask(mask).to_event(3),
                        obs=z)

        emission_probs = self.emitter(z)
        pyro.sample("obs_x", dist.Bernoulli(emission_probs).mask(mask).to_event(3),
                    obs=mini_batch)

    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        T_max = mini_batch.size(1)
        pyro.module("dmm", self)

        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()

        rnn_output, _ = self.rnn(mini_batch_reversed, h_0_contig)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        z_loc, z_scale, z_mat = self.combiner(rnn_output)

        with pyro.poutine.scale(scale=annealing_factor):

            if self.guide_family == "meanfield":
                pyro.sample("z", dist.Normal(z_loc, z_scale).mask(mini_batch_mask.unsqueeze(-1)).to_event(3))
            elif self.guide_family == "markov":
                #print("T_max, z_loc, z_scale, z_mat",T_max, z_loc.shape, z_scale.shape, z_mat.shape)
                hmm = CustomHMM(T_max, z_loc, z_scale, z_mat)
                pyro.sample("z", hmm.to_event(1))


class CustomHMM(TorchDistribution):
    has_rsample = True
    arg_constraints = {}
    support = constraints.real

    def __init__(self, duration, loc, scale, mat):
        self.duration = duration
        self.loc = loc
        self.scale = scale
        self.mat = mat
        self.dim = 4 *  loc.size(-1)

        batch_shape = loc.shape[:-2]
        event_shape = loc.shape[-2:]
        self._log_prob = None
        super().__init__(batch_shape, event_shape, validate_args=False)

    def log_prob(self, value):
        return self._log_prob

    def rsample(self, sample_shape=()):
        eye = torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        trans_matrix = torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        trans_dist = tdist.MultivariateNormal(torch.zeros(self.batch_shape + (self.duration, self.dim),
                                                          dtype=self.loc.dtype, device=self.loc.device),
                                              0.1 * eye)
        trans = matrix_and_mvn_to_gaussian(trans_matrix, trans_dist)

        initial_dist = tdist.MultivariateNormal(torch.zeros(self.batch_shape + (self.dim,),
                                                             dtype=self.loc.dtype, device=self.loc.device), eye)
        init = mvn_to_gaussian(initial_dist)

        eye2 = torch.eye(self.dim // 4, dtype=self.loc.dtype, device=self.loc.device)
        obs_dist = tdist.MultivariateNormal(torch.zeros(self.batch_shape + (self.duration, self.dim // 4),
                                                        dtype=self.loc.dtype, device=self.loc.device),
                                            self.scale.unsqueeze(-1) * eye2)
        obs = matrix_and_mvn_to_gaussian(self.mat, obs_dist)

        factor = trans + obs.condition(self.loc).event_pad(left=self.dim)

        z = _sequential_gaussian_filter_sample(init, factor, sample_shape=sample_shape)

        trans_forward = trans[(slice(0, None), slice(1, None))]
        trans0 = trans[(slice(0, None), 0)]
        num1 = trans_forward.log_density(torch.cat([z[:, :-1, :], z[:, 1:, :]], dim=-1)).sum(-1)
        num2 = (trans0.condition(z[:, 0, :]) + init).event_logsumexp()
        num3 = obs.log_density(torch.cat([z, self.loc], dim=-1)).sum(-1)
        numerator = num1 + num2 + num3

        denom = _sequential_gaussian_tensordot(factor)
        denom = gaussian_tensordot(init, denom, dims=self.dim)
        denom = denom.event_logsumexp()

        self._log_prob = numerator - denom
        return z


# setup, training, and evaluation
def main(args):
    # setup logging
    log = get_logger(args.log)
    log(args)

    data = poly.load_data(poly.JSB_CHORALES)
    training_seq_lengths = data['train']['sequence_lengths']
    training_data_sequences = data['train']['sequences']
    test_seq_lengths = data['test']['sequence_lengths']
    test_data_sequences = data['test']['sequences']
    val_seq_lengths = data['valid']['sequence_lengths']
    val_data_sequences = data['valid']['sequences']
    N_train_data = len(training_seq_lengths)
    N_train_time_slices = float(torch.sum(training_seq_lengths))
    N_mini_batches = int(N_train_data / args.mini_batch_size +
                         int(N_train_data % args.mini_batch_size > 0))

    log("N_train_data: %d     avg. training seq. length: %.2f    N_mini_batches: %d" %
        (N_train_data, training_seq_lengths.float().mean(), N_mini_batches))

    # how often we do validation/test evaluation during training
    val_test_frequency = 10
    # the number of samples we use to do the evaluation
    n_eval_samples = 1

    # package repeated copies of val/test data for faster evaluation
    # (i.e. set us up for vectorization)
    def rep(x):
        rep_shape = torch.Size([x.size(0) * n_eval_samples]) + x.size()[1:]
        repeat_dims = [1] * len(x.size())
        repeat_dims[0] = n_eval_samples
        return x.repeat(repeat_dims).reshape(n_eval_samples, -1).transpose(1, 0).reshape(rep_shape)

    # get the validation/test data ready for the dmm: pack into sequences, etc.
    val_seq_lengths = rep(val_seq_lengths)
    test_seq_lengths = rep(test_seq_lengths)
    val_batch, val_batch_reversed, val_batch_mask, val_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * val_data_sequences.shape[0]), rep(val_data_sequences),
        val_seq_lengths, cuda=args.cuda)
    test_batch, test_batch_reversed, test_batch_mask, test_seq_lengths = poly.get_mini_batch(
        torch.arange(n_eval_samples * test_data_sequences.shape[0]), rep(test_data_sequences),
        test_seq_lengths, cuda=args.cuda)

    # instantiate the dmm
    dmm = DMM(rnn_dropout_rate=args.rnn_dropout_rate, use_cuda=args.cuda, guide_family=args.guide_family)

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    if args.tmc:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC")
        tmc_loss = TraceTMC_ELBO()
        dmm_guide = config_enumerate(dmm.guide, default="parallel", num_samples=args.tmc_num_samples, expand=False)
        svi = SVI(dmm.model, dmm_guide, adam, loss=tmc_loss)
    elif args.tmcelbo:
        if args.jit:
            raise NotImplementedError("no JIT support yet for TMC ELBO")
        elbo = TraceEnum_ELBO()
        dmm_guide = config_enumerate(dmm.guide, default="parallel", num_samples=args.tmc_num_samples, expand=False)
        svi = SVI(dmm.model, dmm_guide, adam, loss=elbo)
    else:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)

    # now we're going to define some functions we need to form the main training loop

    # saves the model and optimizer states to disk
    def save_checkpoint():
        log("saving model to %s..." % args.save_model)
        torch.save(dmm.state_dict(), args.save_model)
        log("saving optimizer states to %s..." % args.save_opt)
        adam.save(args.save_opt)
        log("done saving model and optimizer checkpoints to disk.")

    # loads the model and optimizer states from disk
    def load_checkpoint():
        assert exists(args.load_opt) and exists(args.load_model), \
            "--load-model and/or --load-opt misspecified"
        log("loading model from %s..." % args.load_model)
        dmm.load_state_dict(torch.load(args.load_model))
        log("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        log("done loading model and optimizer states.")

    # prepare a mini-batch and take a gradient step to minimize -elbo
    def process_minibatch(epoch, which_mini_batch, shuffled_indices):
        if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
            # compute the KL annealing factor approriate for the current mini-batch in the current epoch
            min_af = args.minimum_annealing_factor
            annealing_factor = min_af + (1.0 - min_af) * \
                (float(which_mini_batch + epoch * N_mini_batches + 1) /
                 float(args.annealing_epochs * N_mini_batches))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        # compute which sequences in the training set we should grab
        mini_batch_start = (which_mini_batch * args.mini_batch_size)
        mini_batch_end = np.min([(which_mini_batch + 1) * args.mini_batch_size, N_train_data])
        mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
        # grab a fully prepped mini-batch using the helper function in the data loader
        mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
            = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                  training_seq_lengths, cuda=args.cuda)
        # do an actual gradient step
        loss = svi.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                        mini_batch_seq_lengths, annealing_factor)
        # keep track of the training loss
        return loss

    # helper function for doing evaluation
    def do_evaluation():
        # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
        dmm.rnn.eval()

        # compute the validation and test loss n_samples many times
        val_nll = svi.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask,
                                    val_seq_lengths) / float(torch.sum(val_seq_lengths))
        test_nll = svi.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask,
                                     test_seq_lengths) / float(torch.sum(test_seq_lengths))

        # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        dmm.rnn.train()
        return val_nll, test_nll

    # if checkpoint files provided, load model and optimizer states from disk before we start training
    if args.load_opt != '' and args.load_model != '':
        load_checkpoint()

    #################
    # TRAINING LOOP #
    #################
    times = [time.time()]
    for epoch in range(args.num_epochs):
        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
        if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            save_checkpoint()

        # accumulator for our estimate of the negative log likelihood (or rather -elbo) for this epoch
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices for this epoch
        shuffled_indices = torch.randperm(N_train_data)

        # process each mini-batch; this is where we take gradient steps
        for which_mini_batch in range(N_mini_batches):
            epoch_nll += process_minibatch(epoch, which_mini_batch, shuffled_indices)

        # report training diagnostics
        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
            (epoch, epoch_nll / N_train_time_slices, epoch_time))

        # do evaluation on test and validation data and report results
        if val_test_frequency > 0 and epoch > 0 and epoch % val_test_frequency == 0:
            val_nll, test_nll = do_evaluation()
            log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))


# parse command-line arguments and execute the main method
if __name__ == '__main__':
    #assert pyro.__version__.startswith('1.3.1')

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-gf', '--guide-family', type=str, default="markov")
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0003)
    parser.add_argument('-b1', '--beta1', type=float, default=0.96)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=10.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=0.99996)
    parser.add_argument('-wd', '--weight-decay', type=float, default=2.0)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.2)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=0)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str, default='')
    parser.add_argument('-smod', '--save-model', type=str, default='')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--tmc', action='store_true')
    parser.add_argument('--tmcelbo', action='store_true')
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(args)
