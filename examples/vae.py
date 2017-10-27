import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import visdom
import pyro
from pyro.distributions import DiagNormal
from pyro.util import ng_zeros, ng_ones
from pyro.infer import SVI
from pyro.optim import Adam

# for loading and batching MNIST dataset
def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans= transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,
                                              shuffle=False, **kwargs)

    return train_loader, test_loader


# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.view(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.fc21(hidden)
        z_sigma = self.softplus(self.fc22(hidden))
        return z_mu, z_sigma


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        self.fc22 = nn.Linear(hidden_dim, 784)
        # setup the non-linearity
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # send back the mean vector and square root covariance
        # each is of size batch_size x 784
        mu_img = self.fc21(hidden)
        sigma_img = self.softplus(self.fc22(hidden))
        return mu_img, sigma_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 20-dimensional
    # and we use 200 hidden units
    def __init__(self, z_dim=20, hidden_dim=200, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # setup hyperparameters for prior p(z)
        # the type_as ensures we get cuda Tensors if x is on gpu
        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

        # decode the latent code z
        mu_img, sigma_img = self.decoder(z)
        # score against actual images
        pyro.observe("obs", DiagNormal(mu_img, sigma_img), x.view(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder(x)
        # sample the latent code z
        pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_mu, z_sigma = self.encoder(x)
        # sample in latent space
        z = DiagNormal(z_mu, z_sigma).sample()
        # decode the image (not we don't sample in image space)
        mu_img, sigma_img = self.decoder(z)
        return mu_img

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=5000, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=5, type=int, help='how often we evaluate the test set')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    args = parser.parse_args()

    # setup MNIST data loaders
    train_loader, test_loader = setup_data_loaders(use_cuda=args.cuda)

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)

    # setup the optimizer
    optimizer = Adam({"lr": 2.0e-4})

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss="ELBO")

    # setup visdom for visualization
    vis = visdom.Visdom()

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
        normalizer = 28 * 28 * len(train_loader.dataset)
        print("[epoch %03d]  average training loss: %.4f" % (epoch, epoch_loss / normalizer))

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

                # pick a random test image from the first mini-batch and
                # visualize how well we're reconstructing it:1

                if i == 0 :
                    for _ in range(3):
                        test_img = x[np.random.randint(x.size(0)), :, :, :]
                        reco_img = vae.reconstruct_img(test_img)
                        test_img = test_img
                        vis.image(test_img.contiguous().view(28, 28).data.cpu().numpy(),
                                  opts={'caption': 'test image'})
                        vis.image(reco_img.contiguous().view(28, 28).data.cpu().numpy(),
                                  opts={'caption': 'reconstructed image'})

            # report test diagnostics
            normalizer = 28 * 28 * len(test_loader.dataset)
            print("[epoch %03d]  average test loss: %.4f" % (epoch, test_loss / normalizer))


if __name__ == '__main__':
    main()
