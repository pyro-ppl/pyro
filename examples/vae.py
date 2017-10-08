import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import visdom
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
from pyro.util import ng_zeros, ng_ones
from pyro.infer import SVI
from pyro.optim import Adam

# load mnist dataset
root = './data'
download = True
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.MNIST(
    root=root,
    train=True,
    transform=trans,
    download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 128
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=batch_size,
    shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    shuffle=False, **kwargs)


# network
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc21 = nn.Linear(200, 20)
        self.fc22 = nn.Linear(200, 20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), torch.exp(self.fc22(h1))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 2 * 784)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        rv = (self.fc4(h3))

        # reshape to capture mu, sigma params for every pixel
        rvs = rv.view(z.size(0), -1, 2)

        # send back two params
        return rvs[:, :, 0], torch.exp(rvs[:, :, 1])


# create encoder decoder
pt_encode = Encoder()
pt_decode = Decoder()


def model(data):
    # klqp gets called with data.

    # wrap params for use in model -- required
    decoder = pyro.module("decoder", pt_decode)

    # sample from prior
    z_mu, z_sigma = ng_zeros(
        [data.size(0), 20]), ng_ones([data.size(0), 20])

    # sample (retrieve value set by the guide)
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # decode into size of imgx2 for mu/sigma
    img_mu, img_sigma = decoder.forward(z)

    # score against actual images
    pyro.observe("obs", DiagNormal(img_mu, img_sigma), data.view(-1, 784))


def guide(data):
    # wrap params for use in model -- required
    encoder = pyro.module("encoder", pt_encode)

    # use the ecnoder to get an estimate of mu, sigma
    z_mu, z_sigma = encoder.forward(data)

    pyro.sample("latent", DiagNormal(z_mu, z_sigma))


def model_sample():
    # wrap params for use in model -- required
    decoder = pyro.module("decoder", pt_decode)

    # sample from prior
    z_mu, z_sigma = Variable(torch.zeros(
        [1, 20])), Variable(torch.ones([1, 20]))

    # sample
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # decode into size of imgx2 for mu/sigma
    img_mu, img_sigma = decoder.forward(z)

    # score against actual images
    return img_mu


adam = Adam({"lr": 0.0001})
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
