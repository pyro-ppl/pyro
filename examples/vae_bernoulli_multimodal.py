import argparse
import torch
import pyro
from torch.autograd import Variable
from pyro.infer.kl_qp import KL_QP
from pyro.infer.abstract_infer import lw_expectation
from pyro.distributions import DiagNormal, Normal, Bernoulli, Categorical
from torch import nn

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import visdom

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
        self.fc1_c = nn.Linear(10, 200)
        self.fc21 = nn.Linear(200, 20)
        self.fc22 = nn.Linear(200, 20)
        self.relu = nn.ReLU()

    def forward(self, x, cll):
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x) + self.fc1_c(cll))
        return self.fc21(h1), torch.exp(self.fc22(h1))


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc21 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = self.relu(fc1(x))
        alpha_mult = self.softmax(self.fc21(h1))
        return alpha_mult


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 1 * 784)
        self.fc5 = nn.Linear(200, 1 * 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, z):

        h3 = self.relu(self.fc3(z))
        mu_bern = self.sigmoid(self.fc4(h3))
        alpha_mult = self.softmax(self.fc5(h3))
        return mu_bern, alpha_mult


pt_encode = Encoder()
pt_decode = Decoder()


def model(data, cll):

    # wrap params for use in model -- required
    decoder = pyro.module("decoder", pt_decode)

    # sample from prior
    z_mu, z_sigma = Variable(torch.zeros([data.size(0), 20])), Variable(
        torch.ones([data.size(0), 20]))

    # sample
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # decode into size of imgx2 for mu/sigma
    img_mu, alpha_mult = decoder.forward(z)

    # score against actual images
    pyro.observe("obs", Bernoulli(img_mu), data.view(-1, 784))
    pyro.observe("obs_class", Categorical(alpha_mult), cll)


def model_latent(data):

    # wrap params for use in model -- required
    decoder = pyro.module("decoder", pt_decode)

    # sample from prior
    z_mu, z_sigma = Variable(torch.zeros([data.size(0), 20])), Variable(
        torch.ones([data.size(0), 20]))

    # sample
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # decode into size of imgx2 for mu/sigma
    img_mu, alpha_mult = decoder.forward(z)

    # score against actual images
    pyro.observe("obs", Bernoulli(img_mu), data.view(-1, 784))


def guide(data, cll):
    # wrap params for use in model -- required
    encoder = pyro.module("encoder", pt_encode)
    # use the ecnoder to get an estimate of mu, sigma
    z_mu, z_sigma = encoder.forward(data, cll)
    pyro.sample("latent", DiagNormal(z_mu, z_sigma))


def guide_latent(data, cll):
    # wrap params for use in model -- required
    encoder = pyro.module("encoder", pt_encode)
    # use the ecnoder to get an estimate of mu, sigma
    z_mu, z_sigma = encoder.forward(data, cll)
    pyro.sample("latent", DiagNormal(z_mu, z_sigma))


def model_sample():
    # sample from prior
    z_mu, z_sigma = Variable(torch.zeros(
        [1, 20])), Variable(torch.ones([1, 20]))

    # sample
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    # decode into size of imgx1 for mu
    img_mu, alpha = pt_decode.forward(z)
    # score against actual images
    img = pyro.sample("sample_img", Bernoulli(img_mu))
    cll = pyro.sample("sample_cll", Categorical(alpha))
    # return img
    return img, img_mu, cll


def per_param_args(name, param):
    if name == "decoder":
        return {"lr": .0001}
    else:
        return {"lr": .0001}


# or alternatively
adam_params = {"lr": .0001}


inference = KL_QP(model, guide, pyro.optim(optim.Adam, adam_params))
inference_latent = KL_QP(model_latent, guide, pyro.optim(optim.Adam, adam_params))

mnist_data = Variable(train_loader.dataset.train_data.float() / 255.)
mnist_labels = Variable(train_loader.dataset.train_labels)
mnist_size = mnist_data.size(0)
batch_size = 128  # 64


# TODO: batches not necessarily
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
            bs_size = batch_data.size(0)
            batch_class_raw = mnist_labels[batch_start:batch_end]
            batch_class = torch.zeros(bs_size, 10)  # maybe it needs a FloatTensor
            batch_class.scatter_(1, batch_class_raw.data.view(-1, 1), 1)
            batch_class = Variable(batch_class)

            epoch_loss += inference.step(batch_data, batch_class)

        sample, sample_mu, sample_class = model_sample()
        vis.image(batch_data[0].view(28, 28).data.numpy())
        vis.image(sample[0].view(28, 28).data.numpy())
        vis.image(sample_mu[0].view(28, 28).data.numpy())
        print("epoch avg loss {}".format(epoch_loss / float(mnist_size)))

if __name__ == '__main__':
    main()
