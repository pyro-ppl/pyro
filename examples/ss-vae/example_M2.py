
import torch
import pyro
from torch.autograd import Variable
import pyro.distributions as dist
from data_cached import MNISTCached as MNIST
from inference_M2 import SSVAEInfer
from pyro.infer import SVI
from pyro.optim import Adam
from mlp import MLP, Exp, EpsilonScaledSigmoid, EpsilonScaledSoftmax
import torch.nn as nn
import numpy as np
import os.path


class SSVAE(nn.Module):
    """
        This class encapsulates the parameters and functions needed to train a
        semi-supervised variational auto-encoder model on the MNIST image dataset

        :param sup_num: supervised number of examples
                         i.e. how many of the images have supervised labels
        :param output_size: size of the tensor representing the class label (10 for MNIST since
                            we represent the class labels as a one-hot vector with 10 components)
        :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                           since we flatten the images and scale the pixels to be in [0,1])
        :param latent_layer: size of the tensor representing the latent random variable
                             (handwriting style for our MNIST dataset)
        :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                              representing the parameters of the distributions in our model
        :param adam_params: parameters for the adam optimizer used in inference
        :param batch_size: number of images (and labels) to be considered in a batch
        :param epsilon_scale: a small float value used to scale down the output of Softmax and Sigmoid
                              opertations in pytorch for numerical stability
        :param num_epochs: number of epochs to run
        :param use_cuda: use GPUs for faster training
        :param enum_discrete: whether to enumerate the discrete support of the categorical distribution
        :param aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
        :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
        :param logfile: filename for logging the outputs
    """
    def __init__(self, sup_num=3000, output_size=10, input_size=784,
                 latent_layer=20, hidden_layers=(400, 200), adam_params=None,
                 batch_size=100, epsilon_scale=1e-7, num_epochs=100,
                 use_cuda=False, enum_discrete=False, aux_loss=False,
                 aux_loss_multiplier=None, logfile=None):

        super(SSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.sup_num = sup_num
        self.output_size = output_size
        self.input_size = input_size
        self.latent_layer = latent_layer
        self.hidden_layers = hidden_layers
        self.adam_params = adam_params
        self.batch_size = batch_size
        self.epsilon_scale = epsilon_scale
        self.num_epochs = num_epochs
        self.use_cuda = use_cuda
        self.aux_loss = aux_loss
        self.aux_loss_multiplier = aux_loss_multiplier
        self.enum_discrete = enum_discrete
        self.logfile = logfile

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

        # setup the optimizer
        self.optimizer = Adam(self.adam_params)

        # setup the losses to be optimized during inference
        self.setup_losses()

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
            p(z) = normal(0,I)              # handwriting style (latent)
            p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
            p(x|y,z) = bernoulli(mu(y,z))   # an image
            mu is given by a neural network  nn_mu_x

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """

        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        # sample the handwriting style from the constant prior distribution
        const_mu = Variable(torch.zeros([self.batch_size, self.latent_layer]))
        const_sigma = Variable(torch.ones([self.batch_size, self.latent_layer]))
        zs = pyro.sample("z", dist.normal, const_mu, const_sigma)

        # if the label y (which digit to write) is supervised, sample from the
        # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
        alpha_prior = Variable(torch.ones([self.batch_size, self.output_size]) / (1.0 * self.output_size))
        if ys is None:
            ys = pyro.sample("y", dist.categorical, alpha_prior)
        else:
            pyro.observe("y", dist.categorical, ys, alpha_prior)

        # finally, score the image (x) using the handwriting style (z) and
        # the class label y (which digit to write) against the
        # parametrized distribution p(x|y,z) = bernoulli(nn_mu_x(y,z))
        # where nn_mu_x is a neural network
        mu = self.nn_mu_x.forward([zs, ys])
        pyro.observe("x", dist.bernoulli, xs, mu)

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
            q(y|x) = categorical(alpha(x))              # infer digit from an image
            q(z|x,y) = normal(mu(x,y),sigma(x,y))       # infer handwriting style from
                                                          an image and the digit
            mu, sigma are given by a neural network nn_mu_sigma_z
            alpha is given by a neural network nn_alpha_y
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # if the class label (the digit) is not supervised, sample
        # (and score) the digit with the variational distribution
        # q(y|x) = categorical(alpha(x))
        if ys is None:
            alpha = self.nn_alpha_y.forward(xs)
            ys = pyro.sample("y", dist.categorical, alpha)

        # sample (and score) the latent handwriting-style with the variational
        # distribution q(z|x,y) = normal(mu(x,y),sigma(x,y))
        mu, sigma = self.nn_mu_sigma_z.forward([xs, ys])
        zs = pyro.sample("z", dist.normal, mu, sigma)

    def classify(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        alpha = self.nn_alpha_y.forward(xs)

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def setup_networks(self):

        latent_size = self.latent_layer
        hidden_sizes = self.hidden_layers

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter
        # NOTE: we use a customized epsilon-scaled versions of Softmax and
        # Sigmoid operations for numerical stability
        self.nn_alpha_y = MLP([self.input_size] +
                              hidden_sizes +
                              [self.output_size],
                              activation=nn.Softplus,
                              output_activation=EpsilonScaledSoftmax,
                              epsilon_scale=self.epsilon_scale)

        # a split in the final layer's size is used foir multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [latent_size,latent_size]
        # to produce mu and sigma, and apply different activations [None,Exp] on them
        self.nn_mu_sigma_z = MLP([self.input_size + self.output_size] +
                                 hidden_sizes +
                                 [[latent_size, latent_size]],
                                 activation=nn.Softplus,
                                 output_activation=[None, Exp])

        self.nn_mu_x = MLP([latent_size + self.output_size] +
                           hidden_sizes +
                           [self.input_size],
                           activation=nn.Softplus,
                           output_activation=EpsilonScaledSigmoid,
                           epsilon_scale=self.epsilon_scale)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model_classify(self, xs, ys):
        """
            this model is used to add an auxiliary (supervised) loss as described in the
            NIPS 2014 paper by Kingma et al titled
            "Semi-Supervised Learning with Deep Generative Models"
        """

        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # this here is the extra Term to yield an auxiliary loss that we do gradient descend on
        # similar to the NIPS 14 paper (Kingma et al).
        alpha = self.nn_alpha_y.forward(xs)
        pyro.observe("y_aux", dist.categorical, ys, alpha, log_pdf_mask=self.aux_loss_multiplier)

    def guide_classify(self, xs, ys):
        """
            dummy guide function to accompany model_classify in inference
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

    def setup_losses(self):
        # set up the losses for inference
        loss_observed = SVI(self.model, self.guide, self.optimizer, loss="ELBO")

        # setting the enum_discrete parameter builds the loss as a sum by enumerating each class label
        # for the discrete categorical distribution in the model
        loss_latent = SVI(self.model, self.guide, self.optimizer, loss="ELBO", enum_discrete=self.enum_discrete)

        # build a list of all losses considered
        self.losses = [loss_observed, loss_latent]

        # whether the loss should be used for the supervised part (True)
        # of the data or the un-supervised part (False) of the data
        self.is_supervised_loss = [True, False]

        if self.aux_loss:
            loss_aux = SVI(self.model_classify, self.guide_classify, self.optimizer, loss="ELBO")
            self.losses.append(loss_aux)
            self.is_supervised_loss.append(True)

    def save_checkpoint(self, epoch, valid_acc, test_acc, post="last"):
        """
            helper function for saving the state of the SS-VAE module as checkpoints
            this is called by inference after every epoch
        :param epoch: current epoch (int)
        :param valid_acc: current validation accuracy
        :param test_acc: current testing accuracy
        :param post: "best" or "last" to signify if the checkpoint is saving the
                     current best w.r.t validation accuracy or just the state from the
                     last epoch
        :return: None
        """
        assert post in ["last", "best"], "invalid 'post' parameter to save_checkpoint"

        # hard-coded local directory for checkpoints
        dirname = './checkpoints'

        # make the directory if it doesn't exist
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # build the state to be logged - book-keeping the epoch, accuracies,
        # random states in numpy, torch and torch.cuda,
        # the state of this module and the optimizer
        state = {
            'epoch': epoch + 1,
            'ss_vae': self.state_dict(),
            'optimizer': self.optimizer.get_state(),
            'rand_state': torch.get_rng_state(),
            'rand_cuda_state': torch.cuda.get_rng_state(),
            'numpy_rand_state': np.random.get_state(),
            'valid_acc': valid_acc,
            'test_acc': test_acc
        }

        # derive the filename from the logfile
        filename = "ss_vae_{}_{}".format(os.path.basename(self.logfile), post)
        filename = os.path.join(dirname, filename)

        # save the checkpoint file
        torch.save(state, filename)

    def load_checkpoint_randomness(self):
        """
            load the randomness from a checkpoint
            this should be delayed until data loads and gets cached
            using a pre-determined seed
        :return: None
        """
        if self.checkpoint is not None:
            np.random.set_state(self.checkpoint['numpy_rand_state'])
            torch.set_rng_state(self.checkpoint['rand_state'])
            torch.cuda.set_rng_state(self.checkpoint['rand_cuda_state'])

    def load_checkpoint(self, filename):
        self.checkpoint = torch.load(filename)
        self.optimizer.set_state(self.checkpoint['optimizer'])
        self.load_state_dict(self.checkpoint['ss_vae'])
        start_epoch = self.checkpoint['epoch']
        # do less num epochs
        self.num_epochs -= start_epoch

    def optimize(self):
        """
            this function runs the inference
        """
        try:
            # setup the logger if a filename is provided
            logger = None if self.logfile is None else open(self.logfile, "w")

            # setup the inference with appropriate data and losses
            self.inference = SSVAEInfer(MNIST, self.batch_size, self.losses, self.is_supervised_loss,
                                        self.classify, sup_num=self.sup_num, use_cuda=self.use_cuda,
                                        logger=logger)
            # run the inference
            self.inference.run(num_epochs=self.num_epochs, ss_vae_module=self)
        finally:
            # close the logger file object if opened
            if self.logfile is not None:
                logger.close()


def set_seed(seed, use_cuda):
    """
        setting the seed for controlling randomness in this example
    :param seed: seed value (int)
    :param use_cuda: set the random seed for torch.cuda or not
    :return: None
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if use_cuda:
            torch.cuda.manual_seed(seed)


def main(args):
    if args.use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    set_seed(args.seed, args.use_cuda)
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    ss_vae = SSVAE(sup_num=args.sup_num, latent_layer=args.latent_layer,
                   hidden_layers=args.hidden_layers, adam_params=adam_params,
                   batch_size=args.batch_size, epsilon_scale=args.epsilon_scale,
                   num_epochs=args.num_epochs, use_cuda=args.use_cuda,
                   enum_discrete=args.enum_discrete, aux_loss=args.aux_loss,
                   aux_loss_multiplier=args.aux_loss_multiplier, logfile=args.logfile)
    ss_vae.optimize()


EXAMPLE_RUN = "example run: python example_M2.py --seed 0 -cuda -ne 2 --aux-loss -alm 300 -enum -sup 3000 " \
              "-ll 20 -hl 400 200 -lr 0.001 -b1 0.95 -bs 500 -eps 1e-7 -log ./tmp.log"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SS-VAE model inference\n{}".format(EXAMPLE_RUN))

    parser.add_argument('-cuda', '--use-cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-ne', '--num-epochs', default=100, type=int,
                        help="number of epochs to run")
    parser.add_argument('--aux-loss', action="store_true",
                        help="whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)")
    parser.add_argument('-alm', '--aux-loss-multiplier', default=0.1, type=float,
                        help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum-discrete', action="store_true",
                        help="whether to enumerate the discrete support of the categorical distribution"
                             "while computing the ELBO loss")
    parser.add_argument('-sup', '--sup-num', required=True,
                        type=float, choices=[100, 600, 1000, 3000],
                        help="supervised amount of the data i.e. "
                             "how many of the images have supervised labels")
    parser.add_argument('-ll', '--latent-layer', default=20, type=int,
                        help="size of the tensor representing the latent random "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-hl', '--hidden-layers', nargs='+', default=[400, 200], type=int,
                        help="a tuple (or list) of MLP layers to be used in the neural networks "
                             "representing the parameters of the distributions in our model")
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-b1', '--beta-1', default=0.9, type=float,
                        help="beta-1 parameter for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=100, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('-eps', '--epsilon-scale', default=1e-7, type=float,
                        help="a small float value used to scale down the output of Softmax "
                             "and Sigmoid opertations in pytorch for numerical stability")
    parser.add_argument('-log', '--logfile', default=None, type=str,
                        help="filename for logging the outputs")
    parser.add_argument('--seed', default=None, type=int,
                        help="seed for controlling randomness in this example")

    args = parser.parse_args()

    main(args)