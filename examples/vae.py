import argparse
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

# load MNIST dataset
root = './data'
download = True
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

# specify the mini-batch size
batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                           shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,
                                          shuffle=False, **kwargs)


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
        self.relu = nn.ReLU()

    def forward(self, x):
        # define the forward computation
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x))
        # we return a mean vector and a (positive) square root covariance
        # each of length z_dim
        return self.fc21(h1), torch.exp(self.fc22(h1))


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        # setup the four linear transformations used
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 2 * 784)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 2 * 784)
        # setup the two non-linearities used
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        # define the forward computation
        output = self.fc4(self.relu(self.fc3(z)))
        # reshape output to get mu, sigma params for every pixel
        likelihood_params = output.view(z.size(0), -1, 2)
        # send back the mean vector and square root covariance
        # each is of length 784
        return likelihood_params[:, :, 0], torch.exp(likelihood_params[:, :, 1])


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 20-dimensional and we use 200 hidden units
    def __init__(self, z_dim=20, hidden_dim=200, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, data):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # setup hyperparameters for prior p(z)
        z_mu = ng_zeros([data.size(0), self.z_dim], type_as=data.data)
        z_sigma = ng_ones([data.size(0), self.z_dim], type_as=data.data)
        # sample from prior (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

        # decode z
        img_mu, img_sigma = self.decoder(z)
        # score against actual images
        pyro.observe("obs", DiagNormal(img_mu, img_sigma), data.view(-1, 784))

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, data):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder(data)
        # sample the latent code z
        pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # define a helper to sample from generative model
    def model_sample(self):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # setup hyperparameters for prior p(z)
        z_mu, z_sigma = ng_zeros([1, self.z_dim]), ng_ones([1, self.z_dim])
        if self.use_cuda:
            z_mu, z_sigma = z_mu.cuda(), z_sigma.cuda()
        # sample from prior (value will be sampled by guide in computing the ELBO)
        z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

        # decode z
        img_mu, img_sigma = self.decoder(z)
        # return the mean vector img_mu
        return img_mu


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('-tf', '--test-frequency', default=3, type=int, help='how often we evaluate the test set')
    parser.add_argument('--cuda', action='store_true', default=False, help='whether to use cuda')
    args = parser.parse_args()

    # setup the VAE
    vae = VAE(use_cuda=args.cuda)

    # setup the optimizer
    optimizer = Adam({"lr": 0.0001})

    # setup the inference algorithm
    svi = SVI(vae.model, vae.guide, optimizer, loss="ELBO")

    # setup visdom for visualization
    vis = visdom.Visdom()

    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.
        # do a training epoch
        for _, (x, _) in enumerate(train_loader):
            if args.cuda:
                x = x.cuda()
            x = Variable(x.view(-1, 784))
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)

        # report training diagnostics
        print("[epoch %03d]  average training loss: %.4f" % (epoch, epoch_loss / len(train_loader.dataset)))

        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.
            # compute the loss over the entire test set
            for _, (x, _) in enumerate(test_loader):
                if args.cuda:
                    x = x.cuda()
                x = Variable(x.view(-1, 784))
                # do ELBO gradient and accumulate loss
                test_loss += svi.evaluate_loss(x)
            # sample an image and visualize
            sample = vae.model_sample()
            vis.image(sample[0].contiguous().view(28, 28).data.cpu().numpy())

            # report test diagnostics
            print("[epoch %03d]  average test loss: %.4f" % (epoch, test_loss / len(test_loader.dataset)))


if __name__ == '__main__':
    main()
