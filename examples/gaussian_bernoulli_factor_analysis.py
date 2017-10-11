import argparse

import numpy as np
import torch
import torchvision.datasets as dset
from torch.autograd import Variable

import pyro
from pyro.distributions import Bernoulli
from pyro.distributions import DiagNormal
from pyro.infer import SVI
from pyro.optim import Adam

mnist = dset.MNIST(
    root='./',
    train=True,
    transform=None,
    target_transform=None,
    download=True)
print('dataset loaded')
sigmoid = torch.nn.Sigmoid()
softplus = torch.nn.Softplus()


def factor_analysis_model(i, data):
    """
    Factor analysis with a Bernoulli observation model.
    P(x,z,w) = P(x|z,w) P(w) P(z)
    with P(x|z,w) = Bernoulli(mu_beta)
    P(z) = Normal(0,1)
    P(w) = Normal(0,1)
    mu_beta = sigmoid(z * w)
    """
    dim_z = 20
    dim_o = data.size(1)
    nr_samples = data.size(0)

    def sub_model(data, weight):
        mu_latent = Variable(torch.zeros(nr_samples, dim_z))
        sigma_latent = Variable(torch.ones(mu_latent.size()))
        z = pyro.sample("embedding_of_chunk_" + str(i), DiagNormal(mu_latent, sigma_latent))

        mean_observations = pyro.param("observation_mean", Variable(torch.zeros(1, dim_o), requires_grad=True))

        # coordinate times factors yields an activation
        mean_beta_activation = z.mm(weight) + mean_observations.repeat(nr_samples, 1)

        # use sigmoid as link function between the Gaussian activation and the Bernoulli variable
        beta = sigmoid(mean_beta_activation)
        # observe with the Bernoulli
        pyro.observe("obs_" + str(i), Bernoulli(beta), data)

    # global variables are sampled once
    mu_w = Variable(torch.ones(dim_z, dim_o), requires_grad=False)
    sigma_w = Variable(torch.ones(dim_z, dim_o), requires_grad=False)
    weight = pyro.sample("factor_weight", DiagNormal(mu_w, sigma_w))

    # loop over all data and sample the local variables (coordinates/embeddings) for each datum given global variable
    sub_model(data, weight)


def factor_analysis_guide(i, data):
    dim_z = 20
    dim_o = data.size(1)
    nr_samples = data.size(0)

    def inference_model(data):
        mu_q_z = Variable(torch.zeros(nr_samples, dim_z), requires_grad=True)
        log_sigma_q_z = Variable(torch.zeros(mu_q_z.size()), requires_grad=True)

        # parameters for approximate posteriors to the distributions of the embeddings
        guide_mu_z = pyro.param("embedding_posterior_mean_", mu_q_z)
        guide_log_sigma_q_z = pyro.param("embedding_posterior_log_sigma_", log_sigma_q_z)
        guide_sigma_z = torch.exp(guide_log_sigma_q_z)  # * 1e-5

        # sample from approximate posteriors for embeddings
        pyro.sample("embedding_of_chunk_" + str(i), DiagNormal(guide_mu_z, guide_sigma_z))  # z_q

    mu_q_w = Variable(torch.zeros(dim_z, dim_o), requires_grad=True)
    log_sigma_q_w = Variable(torch.zeros(dim_z, dim_o), requires_grad=True)

    # parameters for approximate posteriors to the distribution of the factor weights
    guide_mu_q_w = pyro.param("factor_weight_mean", mu_q_w)
    guide_log_sigma_q_w = log_sigma_q_w - 1e5  # pyro.param("factor_weight_log_sigma", log_sigma_q_w)
    guide_sigma_q_w = torch.exp(guide_log_sigma_q_w)
    guide_sigma_q_w = softplus(guide_sigma_q_w)

    # sample from approximate posterior for weights
    pyro.sample("factor_weight", DiagNormal(guide_mu_q_w, guide_sigma_q_w))  # w_q

    # loop over all data and infer local variables
    inference_model(data)


dat = mnist.train_data
mnist_size = dat.size(0)
m_data = dat.view(mnist_size, -1)
mnist_data = Variable(m_data).float() / 255.
nr_samples = mnist_data.size(0)
batch_size = 10

all_batches = np.arange(0, mnist_size, batch_size)

if all_batches[-1] != mnist_size:
    all_batches = list(all_batches) + [mnist_size]

adam = Adam({"lr": 0.01})
grad_step = SVI(factor_analysis_model, factor_analysis_guide, adam, loss="ELBO")


# apply it to minibatches of data by hand:
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
            epoch_loss += grad_step.step(ix, batch_data)

        print("epoch avg loss {}".format(epoch_loss / float(mnist_size)))


if __name__ == '__main__':
    main()
