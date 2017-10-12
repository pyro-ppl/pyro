"""
An implementation of a Deep Markov Model in Pyro based on reference [1].
This is essentially the DKS variant outlined in the paper. The primary difference
between this implementation and theirs is that in our version any KL divergence terms
in the ELBO are estimated via sampling, while they make use of the analytic formulae. We
also illustrate using normalizing flows in the variational distribution (in which case
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
import pyro.distributions as dist
from pyro.util import ng_ones, zeros
import torch.nn as nn
from pyro.distributions.transformed_distribution import InverseAutoregressiveFlow
from pyro.distributions.transformed_distribution import TransformedDistribution
import six.moves.cPickle as pickle
import polyphonic_data_loader as poly
from pyro.infer import SVI
from pyro.optim import ClippedAdam
from os.path import join, dirname
import logging
import cloudpickle
import os.path


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
        """
        Given the latent z at a particular time step t we return the vector of probabilities
        that parameterizes the bernoulli distribution p(x_t|z_t)
        """
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
        """
        Given the latent z_{t-1} corresponding to the time step t-1 we return the mean and sigma vectors
        that parameterize the (diagonal) gaussian distribution p(z_t | z_{t-1})
        """
        g = self.sigmoid(self.lin_g2(self.relu(self.lin_g1(z))))
        h = self.lin_h2(self.relu(self.lin_h1(z)))
        mu = (ng_ones(g.size()).type_as(g) - g) * self.lin_mu(z) + g * h
        sigma = self.softplus(self.lin_sig(self.relu(h)))
        return mu, sigma


class Combiner(nn.Module):
    """
    Parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block of the
    guide (i.e. the variational distribution). The dependence on x_{t:T} is through the
    hidden state of the RNN (see the pytorch module `rnn` below)
    """
    def __init__(self, z_dim, rnn_dim):
        super(Combiner, self).__init__()
        self.lin_z = nn.Linear(z_dim, rnn_dim)
        self.lin_mu = nn.Linear(rnn_dim, z_dim)
        self.lin_sig = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z, h_rnn):
        """
        Given the latent z at at a particular time step t-1 as well as the hidden state of
        the RNN h(x_{t:T}) we return the mean and sigma vectors that parameterize the (diagonal)
        gaussian distribution p(z_t | z_{t-1})
        """
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
        # instantiate pytorch modules used in the model and guide below
        self.emitter = Emitter(input_dim, z_dim, emission_dim)
        self.trans = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=args.rnn_num_layers,
                          dropout=args.rnn_dropout_rate)

        # if we're using normalizing flows, instantiate those too
        self.iafs = [InverseAutoregressiveFlow(z_dim, 100) for _ in range(args.num_iafs)]
        # make sure each iaf is an attribute of dmm so that pytorch module logic knows about it
        map(lambda i: self.__setattr__('iaf_%d' % i, self.iafs[i]), range(args.num_iafs))

        self.args = args
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim

        # if on gpu cuda-ize all pytorch (sub)modules
        if args.cuda:
            self.cuda()

    # the model p(x|z)p(z)
    def model(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = np.max(mini_batch_seq_lengths)

        # register all pytorch (sub)modules with pyro
        pyro.module("dmm", self)

        # define a (trainable) parameter z_0 that helps define the probability distribution p(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        z_prev = pyro.param("z_0", zeros(self.z_dim, type_as=mini_batch.data))

        # sample the latents z and observed x's one time step at a time
        for t in range(1, T_max + 1):
            # the next three lines of code sample z_t ~ p(z_t | z_{t-1})
            # note that log_pdf_mask takes care of both
            # (i)  KL annealing; and
            # (ii) raggedness in the observed data (i.e. different sequences in the mini-batch
            #      have different lengths)
            z_mu, z_sigma = self.trans(z_prev)
            z_t = pyro.sample("z_%d" % t, dist.DiagNormal(z_mu, z_sigma),
                              log_pdf_mask=annealing_factor * mini_batch_mask[:, t - 1:t])
            # the next three lines observe x_t according to the distribution p(x_t|z_t)
            z_mu, z_sigma = self.trans(z_prev)
            emission_probs_t = self.emitter(z_t)
            pyro.observe("obs_x_%d" % t, dist.bernoulli, mini_batch[:, t - 1, :], emission_probs_t,
                         log_pdf_mask=mini_batch_mask[:, t - 1:t])
            # the lqtent sampled at this time step will be conditioned upon in the next time step
            z_prev = z_t

    # the guide q(z|x) (i.e. the variational distribution)
    def guide(self, mini_batch, mini_batch_reversed, mini_batch_mask,
              mini_batch_seq_lengths, annealing_factor=1.0):

        # this is the number of time steps we need to process in the mini-batch
        T_max = np.max(mini_batch_seq_lengths)
        # register all pytorch (sub)modules with pyro
        pyro.module("dmm", self)

        # define a parameter for the initial state of the rnn
        h_0 = pyro.param("h_0", zeros(self.rnn.num_layers, 1, self.rnn_dim, type_as=mini_batch.data))
        # if on gpu we need the fully broadcast view of h_0 in cuda memory
        if self.args.cuda:
            h_0 = h_0.expand(*[h_0.data.shape[0], mini_batch.data.shape[0], h_0.data.shape[2]]).contiguous()

        # push the observed x's through the rnn; rnn_output contains the hidden at each time step
        rnn_output, _ = self.rnn(mini_batch_reversed, h_0)
        # reverse the time-ordering in the hidden state and un-pack it
        rnn_output = poly.pad_and_reverse(rnn_output, mini_batch_seq_lengths)
        # define a (trainable) parameter z_0 that helps define the probability distribution p(z_1)
        # (since for t = 1 there are no previous latents to condition on)
        z_prev = pyro.param("z_q_0", zeros(self.z_dim, type_as=mini_batch.data))

        # sample the latents z one time step at a time
        for t in range(1, T_max + 1):
            # the next two lines assemble the distribution q(z_t | z_{t-1}, x_{t:T})
            z_mu, z_sigma = self.combiner(z_prev, rnn_output[:, t - 1, :])
            z_dist = dist.DiagNormal(z_mu, z_sigma)

            # if we are including normalizing flows, we register each normalizing flow module with pyro
            # and replace z_dist with a transformed distribution (so that z_dist above is the base distribution
            # of the normalizing flow and not q(z_t|...) itself)
            if args.num_iafs > 0:
                map(lambda i: pyro.module("iaf_%d" % i, self.iafs[i]), range(args.num_iafs))
                z_dist = TransformedDistribution(z_dist, self.iafs)
            # sample z_t from the distribution z_dist
            z_t = pyro.sample("z_%d" % t, z_dist,
                              log_pdf_mask=annealing_factor * mini_batch_mask[:, t - 1:t])
            # the lqtent sampled at this time step will be conditioned upon in the next time step
            z_prev = z_t


# setup, training, and evaluation
def main(num_epochs=5000, learning_rate=0.0008, beta1=0.9, beta2=0.999,
         clip_norm=20.0, lr_decay=1.0, weight_decay=0.10,
         mini_batch_size=20, annealing_epochs=1000,
         minimum_annealing_factor=0.0, rnn_dropout_rate=0.1,
         rnn_num_layers=1, log='dmm.log', cuda=False, num_iafs=0,
         load_opt='', load_model='', checkpoint_freq=0,
         save_opt='', save_model=''):

    # XXX pas what's going on here
    # ensure ints
    num_epochs = int(num_epochs)
    mini_batch_size = int(mini_batch_size)
    annealing_epochs = int(annealing_epochs)

    input_dim = 88
    z_dim = 100
    transition_dim = 200
    emission_dim = 100
    rnn_dim = 600
    val_test_frequency = 30
    expensive_val_test_points = [num_epochs - 1, num_epochs - 1501]
    n_eval_samples_inner = 1
    n_eval_samples_outer = 10

    # setup logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s', filename=args.log, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    def log(s):
        logging.info(s)

    log(args)

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

    # package repeated copies of val/test data for faster evaluation
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

    # instantiate the dmm
    dmm = DMM(input_dim, z_dim, emission_dim, transition_dim, rnn_dim, args)

    # setup optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta1, args.beta2),
                   "clip_norm": args.clip_norm, "lrd": args.lr_decay,
                   "weight_decay": args.weight_decay}
    adam = ClippedAdam(adam_params)

    # setup inference algorithm
    elbo = SVI(dmm.model, dmm.guide, adam, "ELBO", trace_graph=False)
    # by default the KL annealing factor is unity
    annealing_factor = 1.0
    expensive_evaluations = []

    # if provided load model and optimizer states from checkpoints
    if args.load_opt != '' and args.load_model !='':
        assert os.path.exists(args.load_opt) and os.path.exists(args.load_model)
        log("loading model from %s..." % args.load_model)
	dmm.load_state_dict(torch.load(args.load_model))
        log("loading optimizer states from %s..." % args.load_opt)
        adam.load(args.load_opt)
        log("done loading model and optimizer states.")

    # training loop
    for epoch in range(args.num_epochs):
        epoch_nll = 0.0
        # prepare mini-batch subsampling indices
        shuffled_indices = np.arange(N_train_data)
        np.random.shuffle(shuffled_indices)

        # if specified, save model and optimizer states to disk every checkpoint_freq epochs
	if args.checkpoint_freq > 0 and epoch > 0 and epoch % args.checkpoint_freq == 0:
            log("saving model to %s..." % args.save_model)
 	    torch.save(dmm.state_dict(), args.save_model)
            log("saving optimizer states to %s..." % args.save_opt)
	    adam.save(args.save_opt)
            log("done loading model and optimizer states from disk.")

	# process each mini-batch
        for which in range(N_mini_batches):
            if args.annealing_epochs > 0 and epoch < args.annealing_epochs:
                # compute the KL annealing factor approriate for the current mini-batch in the current epoch
                min_af = args.minimum_annealing_factor
                annealing_factor = min_af + (1.0 - min_af) * \
                    (float(which + epoch * N_mini_batches + 1) /
                     float(args.annealing_epochs * N_mini_batches))

            # compute which mini-batch indices we should grab
            mini_batch_start = (which * args.mini_batch_size)
            mini_batch_end = np.min([(which + 1) * args.mini_batch_size, N_train_data])
            mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
            # grab the actual mini-batch using the helper function in the data loader
            mini_batch, mini_batch_reversed, mini_batch_mask, mini_batch_seq_lengths \
                = poly.get_mini_batch(mini_batch_indices, training_data_sequences,
                                      training_seq_lengths, cuda=cuda)
            # do an actual gradient step
            loss = elbo.step(mini_batch, mini_batch_reversed, mini_batch_mask,
                             mini_batch_seq_lengths, annealing_factor)
            # keep track of the training loss
            epoch_nll += loss

        times.append(time.time())
        epoch_time = times[-1] - times[-2]
        log("[training epoch %04d]  %.4f \t\t\t\t(dt = %.3f sec)" %
            (epoch, epoch_nll / N_train_time_slices, epoch_time))

        # helper function for doing evaluation
        def do_evaluation(n_samples):
            # put the RNN into evaluation mode (i.e. turn off drop-out if applicable)
            dmm.rnn.eval()
            val_nlls, test_nlls = [], []

            # compute the validation and test loss n_samples many times
            for _ in range(n_samples):
                val_nll = elbo.evaluate_loss(val_batch, val_batch_reversed, val_batch_mask,
                                             val_seq_lengths) / np.sum(val_seq_lengths)
                test_nll = elbo.evaluate_loss(test_batch, test_batch_reversed, test_batch_mask,
                                              test_seq_lengths) / np.sum(test_seq_lengths)
                val_nlls.append(val_nll)
                test_nlls.append(test_nll)

            # put the RNN back into training mode (i.e. turn on drop-out if applicable)
            dmm.rnn.train()
            val_nll, test_nll = np.mean(val_nlls), np.mean(test_nlls)
            return val_nll, test_nll

        if epoch % val_test_frequency == 0 and epoch > 0:
            val_nll, test_nll = do_evaluation(n_samples=1)
            log("[val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))

        if epoch in expensive_val_test_points:
            val_nll, test_nll = do_evaluation(n_samples=n_eval_samples_outer)
            log("[EXPENSIVE val/test epoch %04d]  %.4f  %.4f" % (epoch, val_nll, test_nll))
            expensive_evaluations.append((val_nll, test_nll))

    return expensive_evaluations


# parse command-line arguments and execute the main method
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, default=5000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0008)
    parser.add_argument('-b1', '--beta1', type=float, default=0.90)
    parser.add_argument('-b2', '--beta2', type=float, default=0.999)
    parser.add_argument('-cn', '--clip-norm', type=float, default=20.0)
    parser.add_argument('-lrd', '--lr-decay', type=float, default=1.0)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.1)
    parser.add_argument('-mbs', '--mini-batch-size', type=int, default=20)
    parser.add_argument('-ae', '--annealing-epochs', type=int, default=1000)
    parser.add_argument('-maf', '--minimum-annealing-factor', type=float, default=0.0)
    parser.add_argument('-rdr', '--rnn-dropout-rate', type=float, default=0.1)
    parser.add_argument('-rnl', '--rnn-num-layers', type=int, default=1)
    parser.add_argument('-iafs', '--num-iafs', type=int, default=0)
    parser.add_argument('-cf', '--checkpoint-freq', type=int, default=0)
    parser.add_argument('-lopt', '--load-opt', type=str, default='')
    parser.add_argument('-lmod', '--load-model', type=str, default='')
    parser.add_argument('-sopt', '--save-opt', type=str)
    parser.add_argument('-smod', '--save-model', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-l', '--log', type=str, default='dmm.log')
    args = parser.parse_args()

    main(**vars(args))
