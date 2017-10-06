import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import visdom
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal, Bernoulli, Categorical
from pyro.optim import Optimize

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
        # self.exp = nn.Exp()

    def forward(self, x, cll):
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x) + self.fc1_c(cll))
        return self.fc21(h1), torch.exp(self.fc22(h1))


class Encoder_xz(nn.Module):
    def __init__(self):
        super(Encoder_xz, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc21 = nn.Linear(200, 20)
        self.fc22 = nn.Linear(200, 20)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x))
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
        h1 = self.relu(self.fc1(x))
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


class Decoder_xz(nn.Module):
    def __init__(self):
        super(Decoder_xz, self).__init__()
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, 1 * 784)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        mu_bern = self.sigmoid(self.fc4(h3))
        return mu_bern


class Decoder_c(nn.Module):
    def __init__(self):
        super(Decoder_c, self).__init__()
        self.fc3 = nn.Linear(20, 200)
        self.fc5 = nn.Linear(200, 1 * 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        alpha_mult = self.softmax(self.fc5(h3))
        return alpha_mult


pt_encode = Encoder()
pt_encode_xz = Encoder_xz()
pt_decode = Decoder()
pt_decode_c = Decoder_c()
pt_decode_xz = Decoder_xz()


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


def model_xz(data, foo):
    decoder_xz = pyro.module("decoder_xz", pt_decode_xz)
    z_mu, z_sigma = Variable(torch.zeros([data.size(0), 20])), Variable(
        torch.ones([data.size(0), 20]))
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))
    img_mu = decoder_xz.forward(z)
    pyro.observe("obs", Bernoulli(img_mu), data.view(-1, 784))
    return z


def model_c(data, cll):
    z = model_xz(data, None)
    decoder_c = pyro.module("decoder_c", pt_decode_c)
    alpha_mu = decoder_c.forward(z)
    pyro.observe("obs_c", Categorical(alpha_mu), cll)


def guide_latent(data, cll):
    encoder = pyro.module("encoder_xz", pt_encode_xz)
    z_mu, z_sigma = encoder.forward(data)
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))
    return z


def model_sample():
    z_mu, z_sigma = Variable(torch.zeros(
        [1, 20])), Variable(torch.ones([1, 20]))
    z = pyro.sample("latent", DiagNormal(z_mu, z_sigma))

    img_mu = pt_decode_xz.forward(z)
    alpha_mu = pt_decode_c.forward(z)

    img = pyro.sample("sample_img", Bernoulli(img_mu))
    cll = pyro.sample("sample_cll", Categorical(alpha_mu))
    return img, img_mu, cll


def classify(data):
    z = guide_latent(data, None)

    img_mu = pt_decode_xz.forward(z)
    alpha_mu = pt_decode_c.forward(z)

    img = pyro.sample("sample_img", Bernoulli(img_mu))
    cll = pyro.sample("sample_cll", Categorical(alpha_mu))
    return img, img_mu, cll


def per_param_args(name, param):
    if name == "decoder":
        return {"lr": .0001}
    else:
        return {"lr": .0001}


# or alternatively
adam_params = {"lr": .0001}

inference = Optimize(model_xz, guide_latent, torch.optim.Adam, adam_params, loss="ELBO")
inference_c = Optimize(model_c, guide_latent, torch.optim.Adam, adam_params, loss="ELBO")

mnist_data = Variable(train_loader.dataset.train_data.float() / 255.)
mnist_labels = Variable(train_loader.dataset.train_labels)
mnist_size = mnist_data.size(0)
batch_size = 128  # 64

mnist_data_test = Variable(test_loader.dataset.test_data.float() / 255.)
mnist_labels_test_raw = Variable(test_loader.dataset.test_labels)
mnist_labels_test = torch.zeros(mnist_labels_test_raw.size(0), 10)
mnist_labels_test.scatter_(1, mnist_labels_test_raw.data.view(-1, 1), 1)
mnist_labels_test = Variable(mnist_labels_test)

# TODO: batches not necessarily
all_batches = np.arange(0, mnist_size, batch_size)

if all_batches[-1] != mnist_size:
    all_batches = list(all_batches) + [mnist_size]

vis = visdom.Visdom(env='vae_lf_100')

loss_training = []
acc_test = []


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

            if np.mod(ix, 100) == 0:
                epoch_loss += inference_c.step(batch_data, batch_class)
            else:
                epoch_loss += inference.step(batch_data, batch_class)

        sample, sample_mu, sample_class = classify(mnist_data_test)
        acc = torch.sum(sample_class * mnist_labels_test) / \
            float(mnist_labels_test.size(0))  # .cpu().numpy()
        acc_val = acc.data.numpy()[0]
        print('accuracy ' + str(acc_val))
        acc_test.append(acc_val)
        # vis.image(batch_data[0].view(28, 28).data.numpy())
        # vis.image(sample[0].view(28, 28).data.numpy())
        vis.image(sample_mu[0].view(28, 28).data.numpy())  # ,opts=dict({'title': str(sample_class)}))
        vis.line(np.array(acc_test), opts=dict(
            {'title': 'Test Classification Acc. given 100% Tr.-labels'}))
        # vis.image(sample_class.view(1,10).data.numpy())
        print("epoch avg loss {}".format(epoch_loss / float(mnist_size)))


if __name__ == '__main__':
    main()
