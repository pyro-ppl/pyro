import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
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


# choose the latent dimension and number of hidden units to use
z_dim = 20
hidden_dim = 200

# create the encoder and decoder networks
encoder = Encoder(z_dim, hidden_dim)
decoder = Decoder(z_dim, hidden_dim)


def model(data):
    # register PyTorch module with Pyro
    pyro.module("decoder", decoder)

    # setup hyperparameters for prior p(z)
    z_mu, z_sigma = ng_zeros([data.size(0), z_dim]), ng_ones([data.size(0), z_dim])
    # sample from prior (value will be sampled by guide in computing the ELBO)
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # decode z
    img_mu, img_sigma = decoder(z)
    # score against actual images
    pyro.observe("obs", DiagNormal(img_mu, img_sigma), data.view(-1, 784))


def guide(data):
    # register PyTorch module with Pyro
    pyro.module("encoder", encoder)
    # use the encoder to get the parameters used to define q(z|x)
    z_mu, z_sigma = encoder(data)
    # sample the latent code z
    pyro.sample("latent", DiagNormal(z_mu, z_sigma))


def model_sample():
    # register PyTorch module with Pyro
    pyro.module("decoder", decoder)

    # setup hyperparameters for prior p(z)
    z_mu, z_sigma = ng_zeros([1, z_dim]), ng_ones([1, z_dim])
    # sample from prior (value will be sampled by guide in computing the ELBO)
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # decode z
    img_mu, img_sigma = decoder.forward(z)
    # return the mean vector img_mu
    return img_mu


# setup the optimizer
adam = Adam({"lr": 0.0001})

# setup the inference algorithm
svi = SVI(model, guide, adam, loss="ELBO")

# num_steps = 1
mnist_data = Variable(train_loader.dataset.train_data.float() / 255.)
mnist_size = mnist_data.size(0)
batch_size = 256

# TODO: batches not necessary
all_batches = np.arange(0, mnist_size, batch_size)

if all_batches[-1] != mnist_size:
    all_batches = list(all_batches) + [mnist_size]

vis = visdom.Visdom()


def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', nargs='?', default=1000, type=int)
    args = parser.parse_args()
    for i in range(args.num_epochs):
        epoch_loss = 0.
        for ix, batch_start in enumerate(all_batches[:-1]):
            batch_end = all_batches[ix + 1]

            # get batch
            batch_data = mnist_data[batch_start:batch_end]
            epoch_loss += svi.step(batch_data)

            sample = model_sample()
        vis.image(batch_data[0].contiguous().view(28, 28).data.numpy())
        vis.image(sample[0].contiguous().view(28, 28).data.numpy())
        print("epoch avg loss {}".format(epoch_loss / float(mnist_size)))


if __name__ == '__main__':
    main()
