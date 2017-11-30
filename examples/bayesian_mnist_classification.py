import argparse
import torch
from torch.autograd import Variable

from torch import nn
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import visdom

import pyro
from pyro.optim import Adam
from pyro.infer import SVI
from pyro.distributions import Normal, Bernoulli, OneHotCategorical
from pyro import poutine

import pdb as pdb

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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc21 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def init_parameters():
        W1_mu = Variable(torch.ones(mu_latent.size()))
        W1_sigma = Variable(torch.ones(mu_latent.size()))

        b1_mu = Variable(torch.ones(mu_latent.size()))
        b1_sigma = Variable(torch.ones(mu_latent.size()))

        W1 = pyro.parame
        pass

    def forward(self, x):
        x = x.view(-1, 784)
        h1 = self.relu(self.fc1(x))
        alpha_mult = self.softmax(self.fc21(h1))
        return alpha_mult


pt_classify = Classifier()

softplus = nn.Softplus()
softmax = nn.Softmax()


def classifier_forward_simple(X, W_1, b_1, W_out, b_out):
    nr_samples = X.size(0)
    act_h1 = X.mm(W_1) + b_1.repeat(nr_samples,1)
    h1 = softplus(act_h1)
    act_alpha_cat = h1.mm(W_out) + b_out.repeat(nr_samples,1)
    alpha_cat = softmax(act_alpha_cat)#+0.001)
    #pdb.set_trace()
    return alpha_cat


def classifier_forward(X):
    q_mu_W1 = pyro.param("W1_posterior_mu")
    q_log_sigma_W1 = pyro.param("W1_posterior_log_sigma")
    q_sigma_W1 = torch.exp(q_log_sigma_W1)
    W_1 = Normal(q_mu_W1, q_sigma_W1).sample()

    q_mu_b1 = pyro.param("b1_posterior_mu")
    q_log_sigma_b1 = pyro.param("b1_posterior_log_sigma")
    q_sigma_b1 = torch.exp(q_log_sigma_b1)
    b_1 = Normal(q_mu_b1, q_sigma_b1).sample()

    q_mu_W_out = pyro.param("W_out_posterior_mu")
    q_log_sigma_W_out = pyro.param("W_out_posterior_log_sigma")
    q_sigma_W_out = torch.exp(q_log_sigma_W_out)
    W_out = Normal(q_mu_W_out, q_sigma_W_out).sample()

    q_mu_b_out = pyro.param("b_out_posterior_mu")
    q_log_sigma_b_out = pyro.param("b_out_posterior_log_sigma")
    q_sigma_b_out = torch.exp(q_log_sigma_b_out)
    b_out = Normal(q_mu_b_out, q_sigma_b_out).sample()
    alpha_cat = classifier_forward_simple(X, W_1, b_1, W_out, b_out)
    return alpha_cat


def model_predict(X=None, nr_samples=10):
    alpha_all = 0.
    sigma_square_all = 0.
    for iis in range(nr_samples):
      alpha_cat = classifier_forward(X)
      alpha_all += alpha_cat

    alpha_all = alpha_all / nr_samples
    return alpha_all

dim_h1 = 100
def model(X, cll):

    [nr_samples, dim_X] = X.size()
    [nr_samples, dim_classes] = cll.size()

    mu_W1 = Variable(torch.zeros(dim_X, dim_h1))
    sigma_W1 = Variable(torch.ones(mu_W1.size()))
    W_1 = pyro.sample("W_1", Normal(mu_W1, sigma_W1))

    mu_b1 = Variable(torch.zeros(1,dim_h1))
    sigma_b1 = Variable(torch.ones(1,dim_h1))
    b_1 = pyro.sample("b_1", Normal(mu_b1, sigma_b1))

    mu_W_out = Variable(torch.zeros(dim_h1, dim_classes))
    sigma_W_out = Variable(torch.ones(mu_W_out.size()))
    W_out = pyro.sample("W_out", Normal(mu_W_out, sigma_W_out))

    mu_b_out = Variable(torch.zeros(1,dim_classes))
    sigma_b_out = Variable(torch.ones(1,dim_classes))
    b_out = pyro.sample("b_out", Normal(mu_b_out, sigma_b_out))

    alpha_cat = classifier_forward_simple(X, W_1, b_1, W_out, b_out)
    with poutine.scale(None, float(mnist_size)/batch_size):
        pyro.observe('observed_class', OneHotCategorical(alpha_cat), cll)


# def model_sample(data, cll):
#     classifier = pyro.module("classifier", pt_classify)
#     alpha_cat = classifier.forward(data)
#     cll = pyro.sample('observed_class', Categorical(alpha_cat))
#     return cll

def guide(X, cll):

    [nr_samples, dim_X] = X.size()
    [nr_samples, dim_classes] = cll.size()

    mu_W1 = Variable(torch.Tensor(dim_X, dim_h1).normal_(0., 0.02),requires_grad=True)
    log_sigma_W1 = Variable(torch.Tensor(dim_X, dim_h1).normal_(-6., 0.02),requires_grad=True)
    q_mu_W1 = pyro.param("W1_posterior_mu", mu_W1)
    q_log_sigma_W1 = pyro.param("W1_posterior_log_sigma", log_sigma_W1)
    q_sigma_W1 = torch.exp(q_log_sigma_W1)
    W_1 = pyro.sample("W_1", Normal(q_mu_W1, q_sigma_W1))

    mu_b1 = Variable(torch.Tensor(1, dim_h1).normal_(0., 0.02),requires_grad=True)
    log_sigma_b1 = Variable(torch.Tensor(1, dim_h1).normal_(-6., 0.02),requires_grad=True)
    q_mu_b1 = pyro.param("b1_posterior_mu", mu_b1)
    q_log_sigma_b1 = pyro.param("b1_posterior_log_sigma", log_sigma_b1)
    q_sigma_b1 = torch.exp(q_log_sigma_b1)
    b_1 = pyro.sample("b_1", Normal(q_mu_b1, q_sigma_b1))

    mu_W_out = Variable(torch.Tensor(dim_h1, dim_classes).normal_(0., 0.02),requires_grad=True)
    log_sigma_W_out = Variable(torch.Tensor(dim_h1, dim_classes).normal_(-2., 0.02),requires_grad=True)
    q_mu_W_out = pyro.param("W_out_posterior_mu", mu_W_out)
    q_log_sigma_W_out = pyro.param("W_out_posterior_log_sigma", log_sigma_W_out)
    q_sigma_W_out = torch.exp(q_log_sigma_W_out)
    W_out = pyro.sample("W_out", Normal(q_mu_W_out, q_sigma_W_out))

    mu_b_out = Variable(torch.Tensor(1, dim_classes).normal_(0., 0.02),requires_grad=True)
    log_sigma_b_out = Variable(torch.Tensor(1, dim_classes).normal_(-2., 0.02),requires_grad=True)
    q_mu_b_out = pyro.param("b_out_posterior_mu", mu_b_out)
    q_log_sigma_b_out = pyro.param("b_out_posterior_log_sigma", log_sigma_b_out)
    q_sigma_b_out = torch.exp(q_log_sigma_b_out)
    b_out = pyro.sample("b_out", Normal(q_mu_b_out, q_sigma_b_out))
    pass



#inference_opt = KL_QP(model, guide, pyro.optim(Adam, adam_params))

# instantiate optim and inference objects
optim = Adam({"lr": 0.001})
svi = SVI(model, guide, optim, loss="ELBO")

mnist_data = Variable(train_loader.dataset.train_data.float() / 255.).view(-1,784)
mnist_labels_raw = train_loader.dataset.train_labels
mnist_size = mnist_data.size(0)

mnist_labels = torch.zeros(mnist_size,10)
labels_reshaped = mnist_labels_raw.view(-1,1)

mnist_labels.scatter_(1,labels_reshaped,1)
mnist_labels = Variable(mnist_labels)

batch_size = 256  # 64
#pdb.set_trace()


# TODO: batches not necessarily
all_batches = np.arange(0, mnist_size, batch_size)

if all_batches[-1] != mnist_size:
    all_batches = list(all_batches) + [mnist_size]

def main():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', nargs='?', default=1000, type=int)
    args = parser.parse_args()
    for i in range(args.num_epochs):
        epoch_loss = 0.
        pred = 0.
        for ix, batch_start in enumerate(all_batches[:-1]):
            batch_end = all_batches[ix + 1]

            # get batch
            batch_data = mnist_data[batch_start:batch_end]
            batch_class = mnist_labels[batch_start:batch_end]

            batch_loss = svi.step(batch_data, batch_class)
            epoch_loss += batch_loss

            if np.mod(i,10)==0:
              batch_alpha = model_predict(batch_data,20)
              pred += batch_alpha.max(1)[0].sum().data.numpy()[0]

        if np.mod(i,10)==0:
          print("epoch avg accuracy {}".format(pred / float(mnist_size)))
        print("epoch avg loss {}".format(epoch_loss / float(mnist_size)))
        #do the following plot: plot a range over the reals and the gaussians in different colours


if __name__ == '__main__':
    main()
    parameter_dict = pyro.get_param_store()
    w_mu=parameter_dict._params['W1_posterior_mu']
    w_logsigma=parameter_dict._params['W1_posterior_log_sigma']
