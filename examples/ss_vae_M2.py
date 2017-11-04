
import torch
import pyro
import sys
from torch.autograd import Variable
import pyro.distributions as dist
from utils.mnist_cached import MNISTCached, setup_data_loaders
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.nn import ClippedSoftmax, ClippedSigmoid
from utils.custom_mlp import MLP, Exp
from utils.vae_plots import plot_conditional_samples_ssvae, mnist_test_tsne_ssvae
from util import set_seed, print_and_log, mkdir_p
import torch.nn as nn

version_warning = '''
11/02/2017: This example does not work with the release version 0.2 of pytorch.
Please install Pytorch from the latest master branch of pytorch or wait a week for the new release.
This example uses a data loader that requires very recent PyTorch features.
'''
torch_version = pyro.util.parse_torch_version()
if (torch_version < (0, 2, 1) and not torch_version[-1].startswith("+")):
    print(version_warning)
    sys.exit(0)


class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder on the MNIST image dataset

    :param output_size: size of the tensor representing the class label (10 for MNIST since
                        we represent the class labels as a one-hot vector with 10 components)
    :param input_size: size of the tensor representing the image (28*28 = 784 for our MNIST dataset
                       since we flatten the images and scale the pixels to be in [0,1])
    :param z_dim: size of the tensor representing the latent random variable z
                  (handwriting style for our MNIST dataset)
    :param hidden_layers: a tuple (or list) of MLP layers to be used in the neural networks
                          representing the parameters of the distributions in our model
    :param epsilon_scale: a small float value used to scale down the output of Softmax and Sigmoid
                          opertations in pytorch for numerical stability
    :param use_cuda: use GPUs for faster training
    :param aux_loss_multiplier: the multiplier to use with the auxiliary loss
    """
    def __init__(self, output_size=10, input_size=784, z_dim=50, hidden_layers=(500,),
                 epsilon_scale=1e-7, use_cuda=False, aux_loss_multiplier=None):

        super(SSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.epsilon_scale = epsilon_scale
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks()

    def setup_networks(self):

        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        # define the neural networks used later in the model and the guide.
        # these networks are MLPs (multi-layered perceptrons or simple feed-forward networks)
        # where the provided activation parameter is used on every linear layer except
        # for the output layer where we use the provided output_activation parameter
        # NOTE: we use a customized epsilon-scaled versions of Softmax and
        # Sigmoid operations for numerical stability
        self.encoder_y = MLP([self.input_size] + hidden_sizes + [self.output_size],
                             activation=nn.Softplus,
                             output_activation=ClippedSoftmax,
                             epsilon_scale=self.epsilon_scale,
                             use_cuda=self.use_cuda)

        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce mu and sigma, and apply different activations [None,Exp] on them
        self.encoder_z = MLP([self.input_size + self.output_size] +
                             hidden_sizes + [[z_dim, z_dim]],
                             activation=nn.Softplus,
                             output_activation=[None, Exp],
                             use_cuda=self.use_cuda)

        self.decoder = MLP([z_dim + self.output_size] +
                           hidden_sizes + [self.input_size],
                           activation=nn.Softplus,
                           output_activation=ClippedSigmoid,
                           epsilon_scale=self.epsilon_scale,
                           use_cuda=self.use_cuda)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # handwriting style (latent)
        p(y|x) = categorical(I/10.)     # which digit (semi-supervised)
        p(x|y,z) = bernoulli(mu(y,z))   # an image
        mu is given by a neural network  `decoder`

        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        batch_size = xs.size(0)
        with pyro.iarange("independent"):

            # sample the handwriting style from the constant prior distribution
            prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
            prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
            zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma)

            # if the label y (which digit to write) is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)
            alpha_prior = Variable(torch.ones([batch_size, self.output_size]) / (1.0 * self.output_size))
            if ys is None:
                ys = pyro.sample("y", dist.categorical, alpha_prior)
            else:
                pyro.observe("y", dist.categorical, ys, alpha_prior)

            # finally, score the image (x) using the handwriting style (z) and
            # the class label y (which digit to write) against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            mu = self.decoder.forward([zs, ys])
            pyro.observe("x", dist.bernoulli, xs, mu)

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer digit from an image
        q(z|x,y) = normal(mu(x,y),sigma(x,y))       # infer handwriting style from an image and the digit
        mu, sigma are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`
        :param xs: a batch of scaled vectors of pixels from an image
        :param ys: (optional) a batch of the class labels i.e.
                   the digit corresponding to the image(s)
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.iarange("independent"):

            # if the class label (the digit) is not supervised, sample
            # (and score) the digit with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.categorical, alpha)

            # sample (and score) the latent handwriting-style with the variational
            # distribution q(z|x,y) = normal(mu(x,y),sigma(x,y))
            mu, sigma = self.encoder_z.forward([xs, ys])
            zs = pyro.sample("z", dist.normal, mu, sigma)   # noqa: F841

    def classifier(self, xs):
        """
        classify an image (or a batch of images)

        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        # use the trained model q(y|x) = categorical(alpha(x))
        # compute all class probabilities for the image(s)
        alpha = self.encoder_y.forward(xs)

        # get the index (digit) that corresponds to
        # the maximum predicted class probability
        res, ind = torch.topk(alpha, 1)

        # convert the digit(s) to one-hot tensor(s)
        ys = Variable(torch.zeros(alpha.size()))
        ys = ys.scatter_(1, ind, 1.0)
        return ys

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        NIPS 2014 paper by Kingma et al titled
        "Semi-Supervised Learning with Deep Generative Models"
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.iarange("independent"):
            # this here is the extra Term to yield an auxiliary loss that we do gradient descend on
            # similar to the NIPS 14 paper (Kingma et al).
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                pyro.observe("y_aux", dist.categorical, ys, alpha, log_pdf_mask=self.aux_loss_multiplier)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass

    def model_sample(self, ys, batch_size=1):
        # sample the handwriting style from the constant prior distribution
        prior_mu = Variable(torch.zeros([batch_size, self.z_dim]))
        prior_sigma = Variable(torch.ones([batch_size, self.z_dim]))
        zs = pyro.sample("z", dist.normal, prior_mu, prior_sigma)

        # sample an image using the decoder
        mu = self.decoder.forward([zs, ys])
        xs = pyro.sample("sample", dist.bernoulli, mu)
        return xs, mu


def run_inference_for_epoch(data_loaders, losses, periodic_interval_batches):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders["sup"])
    unsup_batches = len(data_loaders["unsup"])
    batches_per_epoch = sup_batches + unsup_batches

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_losses
    epoch_losses_unsup = [0.] * num_losses

    # setup the iterators for training data loaders
    sup_iter = iter(data_loaders["sup"])
    unsup_iter = iter(data_loaders["unsup"])

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0
    for i in range(batches_per_epoch):

        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches

        # extract the corresponding batch
        if is_supervised:
            (xs, ys) = next(sup_iter)
            ctr_sup += 1
        else:
            (xs, ys) = next(unsup_iter)
        xs, ys = Variable(xs), Variable(ys)

        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].step(xs, ys)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(xs)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup


def get_accuracy(data_loader, classifier_fn, batch_size):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []

    # use the appropriate data loader
    for (xs, ys) in data_loader:
        # use classification function to compute all predictions for each batch
        xs, ys = Variable(xs), Variable(ys)
        predictions.append(classifier_fn(xs))
        actuals.append(ys)

    # compute the number of accurate predictions
    accurate_preds = 0
    for pred, act in zip(predictions, actuals):
        for i in range(pred.size(0)):
            v = torch.sum(pred[i] == act[i])
            accurate_preds += (v.data[0] == 10)

    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / (len(predictions) * batch_size)
    return accuracy


def visualize(ss_vae, viz, test_loader):
    if viz is not None:
        plot_conditional_samples_ssvae(ss_vae, viz)
        mnist_test_tsne_ssvae(ssvae=ss_vae, test_loader=test_loader)


def run_inference_ss_vae(args):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """
    if args.use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if args.seed is not None:
        set_seed(args.seed, args.use_cuda)

    viz = None
    if args.visualize:
        from visdom import Visdom
        viz = Visdom()
        mkdir_p("./vae_results")

    # batch_size: number of images (and labels) to be considered in a batch
    ss_vae = SSVAE(z_dim=args.z_dim,
                   hidden_layers=args.hidden_layers,
                   epsilon_scale=args.epsilon_scale,
                   use_cuda=args.use_cuda,
                   aux_loss_multiplier=args.aux_loss_multiplier)

    # setup the optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    optimizer = Adam(adam_params)

    # set up the loss(es) for inference setting the enum_discrete parameter builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    loss_basic = SVI(ss_vae.model, ss_vae.guide, optimizer, loss="ELBO", enum_discrete=args.enum_discrete)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
    if args.aux_loss:
        loss_aux = SVI(ss_vae.model_classify, ss_vae.guide_classify, optimizer, loss="ELBO")
        losses.append(loss_aux)

    try:
        # setup the logger if a filename is provided
        logger = None if args.logfile is None else open(args.logfile, "w")

        data_loaders = setup_data_loaders(MNISTCached, args.use_cuda, args.batch_size, sup_num=args.sup_num)

        # how often would a supervised batch be encountered during inference
        # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
        # until we have traversed through the all supervised batches
        periodic_interval_batches = int(MNISTCached.train_data_size / (1.0 * args.sup_num))

        # number of unsupervised examples
        unsup_num = MNISTCached.train_data_size - args.sup_num

        # initializing local variables to maintain the best validation accuracy
        # seen across epochs over the supervised training set
        # and the corresponding testing set and the state of the networks
        best_valid_acc, corresponding_test_acc = 0.0, 0.0

        # run inference for a certain number of epochs
        for i in range(0, args.num_epochs):

            # get the losses for an epoch
            epoch_losses_sup, epoch_losses_unsup = \
                run_inference_for_epoch(data_loaders, losses, periodic_interval_batches)

            # compute average epoch losses i.e. losses per example
            avg_epoch_losses_sup = map(lambda v: v / args.sup_num, epoch_losses_sup)
            avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)

            # store the loss and validation/testing accuracies in the logfile
            str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
            str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))

            str_print = "{} epoch: avg losses {}".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))

            validation_accuracy = get_accuracy(data_loaders["valid"], ss_vae.classifier, args.batch_size)
            str_print += " validation accuracy {}".format(validation_accuracy)

            # this test accuracy is only for logging, this is not used
            # to make any decisions during training
            test_accuracy = get_accuracy(data_loaders["test"], ss_vae.classifier, args.batch_size)
            str_print += " test accuracy {}".format(test_accuracy)

            # update the best validation accuracy and the corresponding
            # testing accuracy and the state of the parent module (including the networks)
            if best_valid_acc < validation_accuracy:
                best_valid_acc = validation_accuracy
                corresponding_test_acc = test_accuracy

            print_and_log(logger, str_print)

        final_test_accuracy = get_accuracy(data_loaders["test"], ss_vae.classifier, args.batch_size)
        print_and_log(logger, "best validation accuracy {} corresponding testing accuracy {} "
                      "last testing accuracy {}".format(best_valid_acc, corresponding_test_acc, final_test_accuracy))

        # visualize the conditional samples
        visualize(ss_vae, viz, data_loaders["test"])
    finally:
        # close the logger file object if we opened it earlier
        if args.logfile is not None:
            logger.close()


EXAMPLE_RUN = "example run: python example_M2.py --seed 0 -cuda -ne 2 --aux-loss -alm 300 -enum -sup 3000 " \
              "-zd 20 -hl 400 200 -lr 0.001 -b1 0.95 -bs 500 -eps 1e-7 -log ./tmp.log"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SS-VAE model inference\n{}".format(EXAMPLE_RUN))

    parser.add_argument('-cuda', '--use-cuda', action='store_true',
                        help="use GPU(s) to speed up training")
    parser.add_argument('-ne', '--num-epochs', default=1, type=int,
                        help="number of epochs to run")
    parser.add_argument('--aux-loss', action="store_true",
                        help="whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)")
    parser.add_argument('-alm', '--aux-loss-multiplier', default=300, type=float,
                        help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum-discrete', action="store_true",
                        help="whether to enumerate the discrete support of the categorical distribution"
                             "while computing the ELBO loss")
    parser.add_argument('-sup', '--sup-num', default=3000,
                        type=float, help="supervised amount of the data i.e. "
                                         "how many of the images have supervised labels")
    parser.add_argument('-zd', '--z-dim', default=50, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable (handwriting style for our MNIST dataset)")
    parser.add_argument('-hl', '--hidden-layers', nargs='+', default=[500], type=int,
                        help="a tuple (or list) of MLP layers to be used in the neural networks "
                             "representing the parameters of the distributions in our model")
    parser.add_argument('-lr', '--learning-rate', default=0.0003, type=float,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-b1', '--beta-1', default=0.9, type=float,
                        help="beta-1 parameter for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=200, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('-eps', '--epsilon-scale', default=1e-7, type=float,
                        help="a small float value used to scale down the output of Softmax "
                             "and Sigmoid opertations in pytorch for numerical stability")
    parser.add_argument('-log', '--logfile', default="./tmp.log", type=str,
                        help="filename for logging the outputs")
    parser.add_argument('--seed', default=None, type=int,
                        help="seed for controlling randomness in this example")
    parser.add_argument('--visualize', action="store_true",
                        help="use a visdom server to visualize the embeddings")
    args = parser.parse_args()

    # some assertions to make sure that batching math assumptions are met
    assert args.sup_num % args.batch_size == 0, "assuming simplicity of batching math"
    assert MNISTCached.validation_size % args.batch_size == 0, \
        "batch size should divide the number of validation examples"
    assert MNISTCached.train_data_size % args.batch_size == 0, \
        "batch size doesn't divide total number of training data examples"
    assert MNISTCached.test_size % args.batch_size == 0, "batch size should divide the number of test examples"

    run_inference_ss_vae(args)
