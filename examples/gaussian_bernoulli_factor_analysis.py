from pdb import set_trace as bb
import numpy as np
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal
from pyro.distributions import Bernoulli

from pyro.infer.abstract_infer import LikelihoodWeighting, lw_expectation
from pyro.infer.importance_sampling import ImportanceSampling
from pyro.infer.kl_qp import KL_QP

import torchvision.datasets as dset
import torchvision.transforms as transforms
mnist = dset.MNIST(
    root='./',
    train=True,
    transform=None,
    target_transform=None,
    download=True)
print('dataset loaded')

sigmoid = torch.nn.Sigmoid()


def local_model(i, data):
    dim_z = 2
    dim_o = 784
    nr_samples = 1

    nr_data = data.size(0)

    #global variables
    mu_w = Variable(torch.ones(dim_z, dim_o), requires_grad=False)
    log_sigma_w = Variable(torch.ones(dim_z, dim_o), requires_grad=False)
    sigma_w = torch.exp(log_sigma_w)
    weight = pyro.sample("factor_weight", DiagNormal(mu_w, sigma_w))

    def sub_model(datum):
        mu_latent = Variable(torch.ones(nr_samples, dim_z)) * 0.5
        sigma_latent = Variable(torch.ones(mu_latent.size()))
        z = pyro.sample(
            "embedding_of_datum_" +
            str(i),
            DiagNormal(
                mu_latent,
                sigma_latent))
        mean_beta = z.mm(weight)
        beta = sigmoid(mean_beta)
        pyro.observe("obs_" + str(i), Bernoulli(beta), datum)

    for i in range(nr_data):
        sub_model(data[i])

    return z, weight


def local_guide(i, datum):
    dim_z = 2
    dim_o = 784
    nr_samples = 1
    alpha = torch.ones(nr_samples, dim_z) * 0.1
    mu_q_z = Variable(alpha, requires_grad=True)
    log_sigma_q_z = Variable(torch.ones(mu_q_z.size()), requires_grad=True)
    sigma_q_z = torch.exp(log_sigma_q_z)

    mu_q_w = Variable(torch.ones(dim_z, dim_o), requires_grad=True)
    log_sigma_q_w = Variable(torch.ones(dim_z, dim_o), requires_grad=True)

    guide_mu_q_w = pyro.param("factor_weight_mean", mu_q_w)
    guide_log_sigma_q_w = pyro.param("factor_weight_log_sigma", log_sigma_q_w)
    #sigma_q_w = torch.exp(log_sigma_q_w)
    guide_sigma_q_w = torch.exp(guide_log_sigma_q_w)

    guide_mu_z = pyro.param("embedding_posterior_mean_", mu_q_z)

    guide_log_sigma_q_z = pyro.param(
        "embedding_posterior_sigma_", log_sigma_q_z)
    guide_sigma_z = torch.exp(guide_log_sigma_q_z)
    z_q = pyro.sample(
        "embedding_of_datum_" +
        str(i),
        DiagNormal(
            guide_mu_z,
            guide_sigma_z))

    w_q = pyro.sample(
        "factor_weight",
        DiagNormal(
            guide_mu_q_w,
            guide_sigma_q_w))

    return z_q, w_q


#grad_step = ELBo(local_model, local_guide, model_ML=true, optimizer="adam")
adam_params = {"lr": .00000000000001}
adam_optim = pyro.optim(torch.optim.Adam, adam_params)

data = Variable(mnist.train_data).float() / 255.
nr_samples = data.size(0)
nr_epochs = 1000
grad_step = KL_QP(local_model, local_guide, adam_optim)

# apply it to minibatches of data by hand:
for j in range(nr_epochs):
    score = 0
    for i in range(nr_batches):
        score_d = grad_step(i, data[i])
        score += score_d / float(nr_samples)
        print('Local Score ' + str(-score))
    print('Epoch score ' + str(-score))
    # bb()

    #print('starting datum '+str(i))
    # mod_forward=local_model(i,data[i])
    # mod_inv=local_guide(i,data[i])

    # grad_step(i,d)
    # par=pyro._param_store._params['main']['factor_weight_mean']
    # print(str(par.data.numpy()))
    # bb()
    # pyro
