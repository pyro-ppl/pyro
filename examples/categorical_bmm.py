import argparse
import numpy as np
import torch
from torch.nn import Softmax
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
from pyro.distributions import Bernoulli, Categorical
import visdom

from pyro.infer.kl_qp import KL_QP

import torchvision.datasets as dset
import torchvision.transforms as transforms

mnist = dset.MNIST(
    root='./data',
    train=True,
    transform=None,
    target_transform=None,
    download=True)
print('dataset loaded')

softmax = Softmax()

train_loader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])),
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dset.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=128, shuffle=True)


def local_model(i, datum):
    beta = Variable(torch.ones(1, 10)) * 0.1
    cll = pyro.sample("class_of_datum_" + str(i), Categorical(beta))
    mean_param = Variable(torch.zeros(1, 784), requires_grad=True)
    # do MLE for class means
    mu = pyro.param("mean_of_class_" + str(cll[0]), mean_param)
    mu_param = softmax(mu)
    pyro.observe("obs_" + str(i), Bernoulli(mu_param), datum)
    return cll


def local_guide(i, datum):
    alpha = torch.ones(1, 10) * 0.1
    beta_q = Variable(alpha, requires_grad=True)
    beta_param = pyro.param("class_posterior_", beta_q)
    guide_params = softmax(beta_param)
    cll = pyro.sample("class_of_datum_" + str(i), Categorical(guide_params))
    return cll


def inspect_posterior_samples(i):
    cll = local_guide(i, None)
    mean_param = Variable(torch.zeros(1, 784), requires_grad=True)
    # do MLE for class means
    mu = pyro.param("mean_of_class_" + str(cll[0]), mean_param)
    dat = pyro.sample("obs_" + str(i), Bernoulli(mu))
    return dat


optim_fct = pyro.optim(torch.optim.Adam, {'lr': .0001})

inference = KL_QP(local_model, local_guide, optim_fct)

vis = visdom.Visdom()

nr_epochs = 50
# apply it to minibatches of data by hand:

mnist_data = Variable(train_loader.dataset.train_data.float() / 255.)
mnist_labels = Variable(train_loader.dataset.train_labels)
mnist_size = mnist_data.size(0)
batch_size = 1  # 64

all_batches = np.arange(0, mnist_size, batch_size)

if all_batches[-1] != mnist_size:
    all_batches = list(all_batches) + [mnist_size]

def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', type=int, required=True)
    args = parser.parse_args()
    for i in range(args.num_epochs):
        epoch_loss = 0.
        for ix, batch_start in enumerate(all_batches[:-1]):
            batch_end = all_batches[ix + 1]
            batch_data = mnist_data[batch_start:batch_end]
            bs_size = batch_data.size(0)
            batch_class_raw = mnist_labels[batch_start:batch_end]
            batch_class = torch.zeros(bs_size, 10)  # maybe it needs a FloatTensor
            batch_class.scatter_(1, batch_class_raw.data.view(-1, 1), 1)
            batch_class = Variable(batch_class)
            epoch_loss += inference.step(ix, batch_data)

           # optional  visualization!
    #      vis.image(batch_data[0].view(28, 28).data.numpy())
    #      vis.image(sample[0].view(28, 28).data.numpy())
    #      vis.image(sample_mu[0].view(28, 28).data.numpy())
        print("epoch avg loss {}".format(epoch_loss / float(mnist_size)))
if __name__ == '__main__':
    main()
