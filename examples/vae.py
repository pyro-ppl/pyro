import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
import visdom
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.util import ng_zeros, ng_ones
from pyro.nn import ClippedSoftmax, ClippedSigmoid
from utils.vae_plots import plot_llk, mnist_test_tsne
from utils.mnist_cached import MNISTCached as MNIST
from utils.mnist_cached import setup_data_loaders
from utils.custom_mlp import MLP, Exp

fudge = 1e-7

"""
def setup_data_loaders(batch_size=128, use_cuda=False):
    # for loading and batching MNIST dataset
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,
                                              shuffle=False, **kwargs)

    return train_loader, test_loader
"""



# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_layers=[400], use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.setup_networks()

    def setup_networks(self):
        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers

        self.encoder = MLP([784] + hidden_sizes + [[z_dim, z_dim]],
                                 activation=nn.Softplus, output_activation=[None, Exp], use_cuda=self.use_cuda)

        self.decoder = MLP([z_dim] + hidden_sizes + [784],
                           activation=nn.Softplus, output_activation=ClippedSigmoid,
                           epsilon_scale=fudge, use_cuda=self.use_cuda)

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # setup hyperparameters for prior p(z)
        # the type_as ensures we get cuda Tensors if x is on gpu
        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)
        # decode the latent code z
        mu_img = self.decoder(z)
        # score against actual images
        pyro.observe("obs", dist.bernoulli, x.view(-1, 784), mu_img)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder(x)
        # sample the latent code z
        pyro.sample("latent", dist.normal, z_mu, z_sigma)

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_mu, z_sigma = self.encoder(x)
        # sample in latent space
        z = dist.normal(z_mu, z_sigma)
        # decode the image (note we don't sample in image space)
        mu_img = self.decoder(z)
        return mu_img


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=501, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=10, type=int, help='how often we evaluate the test set')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-b1', '--beta1', default=0.95, type=float, help='beta1 adam hyperparameter')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    parser.add_argument('-visdom', '--visdom_flag', default=False, help='Whether plotting in visdom is desired')
    parser.add_argument('-i-tsne', '--tsne_iter', default=20, type=int, help='epoch when tsne visualization runs')
    args = parser.parse_args()


    # setup MNIST data loaders
    # train_loader, test_loader
    train_loader, test_loader = setup_data_loaders(MNIST,use_cuda=args.cuda,batch_size=200)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)

    # setup the optimizer
    adam_args = {"lr": args.learning_rate}# "betas": (args.beta1}#, 0.999)}
    optimizer = Adam(adam_args)

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss="ELBO")

    # setup visdom for visualization
    if args.visdom_flag:
        vis = visdom.Visdom()

    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for _, (x, _) in enumerate(train_loader):
            # if on GPU put mini-batch into CUDA memory
            if args.cuda:
                x = x.cuda()
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # wrap the mini-batch in a PyTorch Variable
                x = Variable(x)
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)

                # pick three random test images from the first mini-batch and
                # visualize how well we're reconstructing them
                if i == 0:
                    if args.visdom_flag:
                        reco_indices = np.random.randint(0, x.size(0), 3)
                        for index in reco_indices:
                            test_img = x[index, :, :, :]
                            reco_img = vae.reconstruct_img(test_img)
                            vis.image(test_img.contiguous().view(28, 28).data.cpu().numpy(),
                                      opts={'caption': 'test image'})
                            vis.image(reco_img.contiguous().view(28, 28).data.cpu().numpy(),
                                      opts={'caption': 'reconstructed image'})

            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test))

        if epoch == args.tsne_iter:
            mnist_test_tsne(vae=vae, test_loader=test_loader)
            plot_llk(np.array(train_elbo), np.array(test_elbo))

    return vae


if __name__ == '__main__':
    model = main()
