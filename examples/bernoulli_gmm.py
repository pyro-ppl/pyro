from pdb import set_trace as bb
import numpy as np
import torch

from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
from pyro.distributions import Bernoulli
import visdom

# from pyro.infer.abstract_infer import LikelihoodWeighting, lw_expectation
# from pyro.infer.importance_sampling import ImportanceSampling
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


def local_model(i, datum):
    beta = Variable(torch.ones(1,1)) * 0.5
    c = pyro.sample("class_of_datum_" + str(i), Bernoulli(beta))
    mean_param = Variable(torch.zeros(784, 1), requires_grad=True)
    # do MLE for class means
    m = pyro.param("mean_of_class_" + str(c[0]), mean_param)

    sigma = Variable(torch.ones(m.size()))
    pyro.observe("obs_" + str(i), DiagNormal(m, sigma), datum)
    return c


def local_guide(i, datum):
    alpha = torch.ones(1,1) * 0.1
    beta_q = Variable(alpha, requires_grad=True)
    guide_params = pyro.param("class_posterior_", beta_q)
    c = pyro.sample("class_of_datum_" + str(i), Bernoulli(guide_params))
    return c


def inspect_posterior_samples(i):
    c = local_guide(i, None)
    mean_param = Variable(torch.zeros(784, 1), requires_grad=True)
    # do MLE for class means
    m = pyro.param("mean_of_class_" + str(c[0]), mean_param)
    sigma = Variable(torch.ones(m.size()))
    dat = pyro.sample("obs_" + str(i), DiagNormal(m, sigma))
    return dat


#grad_step = ELBo(local_model, local_guide, model_ML=true, optimizer="adam")
optim_fct = pyro.optim(torch.optim.Adam, {'lr': .0001})

data = Variable(mnist.train_data).float() / 255.
nr_samples = data.size(0)
grad_step = KL_QP(local_model, local_guide, optim_fct)

d0 = inspect_posterior_samples(0)
d1 = inspect_posterior_samples(1)

vis = visdom.Visdom()

nr_epochs = 50
# apply it to minibatches of data by hand:
for epoch in range(nr_epochs):
    total_loss = 0.
    for i in range(nr_samples):
        # print('starting datum '+str(i))
        # mod_forward=local_model(i,data[i])
        # mod_inv=local_guide(i,data[i])
        loss_sample = grad_step(i, data[i])
        total_loss += loss_sample
        dat = inspect_posterior_samples(i)
        d2 = dat.data.numpy().reshape((28, 28))
        do2 = data[i].view(28, 28).data.numpy()
        # grad_step(i,d)
        #print("loss per sample {}".format(loss_sample))
    vis.image(d2)
    vis.image(do2)
    print("loss per epoch {}".format(total_loss / nr_samples))
