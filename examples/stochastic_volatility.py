import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta, AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


"""
Multivariate Stochastic Volatility Model with constant correlation
See Model 2 on page 365 in https://pdfs.semanticscholar.org/fccc/6f4ee933d4330eabf377c08f8b2650e1f244.pdf
"""
def model(data):
    """
    y = diag(exp(h_t / 2)) * eps_t
    eps ~ Q L_eps rho_t
    h_{i+1} = mu + Phi(h_t - mu) + eta

    We do this in log space to convert multiplicative noise to additive noise
    so we can leverage the GaussianHMM distribution.

    log y_kt = h_kt / 2 + log <L_eps, delta_t>
            ~= h_kt / 2 + gamma_kt where gamma ~ MVN(0. sigma)
    and we moment match to compute epsilon.

    :param data: Tensor of the shape ``(batch, timesteps, returns)``
    :type data: torch.Tensor
    :returns distribution: posterior distribution over states given data
    """
    obs_dim = data.shape[-1]
    hidden_dim = obs_dim
    with pyro.plate(len(data)):
        mu = pyro.param('mu', torch.zeros(hidden_dim))
        L = pyro.param('L', 0.1 * torch.eye(hidden_dim), constraint=constraints.lower_cholesky)
        init_dist = dist.MultivariateNormal(mu, scale_tril=L)

        L_eta = pyro.param('L_eta', 0.4 * torch.eye(hidden_dim), constraint=constraints.lower_cholesky)
        mu_eta = torch.zeros(hidden_dim)
        trans_matrix = pyro.param('phi', 0.5 * torch.ones(hidden_dim))
        # this gives us a zero matrix with phi on the diagonal
        trans_matrix = trans_matrix.diag()
        trans_dist = dist.MultivariateNormal(mu_eta, scale_tril=L_eta)

        mu_gamma = pyro.param('mu_gamma', torch.zeros(obs_dim))
        L_gamma = pyro.param('L_gamma', 0.5 * torch.eye(obs_dim), constraint=constraints.lower_cholesky)
        obs_matrix = torch.eye(hidden_dim, obs_dim)
        # latent state is h_t - mu
        obs_dist = dist.MultivariateNormal(-mu_gamma, scale_tril=L_gamma)

        hmm_dist = dist.GaussianHMM(init_dist, trans_matrix, trans_dist, obs_matrix, obs_dist)
        pyro.sample('obs', hmm_dist, obs=data)

        return hmm_dist.filter(data)


def sequential_model(num_samples=10, timesteps=500, hidden_dim=2, obs_dim=2, init_dist=None):
    """
    Generate data of shape: (samples, timesteps, obs_dim)
    where the generative model is defined by:
        y = exp(h/2) * eps
        h_{t+1} = mu + Phi (h_t - mu) + eta_t
    where eps and eta are sampled iid from a MVN distribution
    """
    ys = []
    mu_trans = torch.zeros(hidden_dim)
    cov_trans = pyro.param('L_eta', 0.2 * torch.eye(hidden_dim, hidden_dim))
    mu_obs = pyro.param('mu_gamma', torch.zeros(obs_dim))
    cov_obs = pyro.param('L_gamma', 0.2 * torch.eye(obs_dim, obs_dim))
    transition = pyro.param('phi', 0.2 * torch.randn(hidden_dim))
    # this is to generate data as the way model 2 does
    # we would use the entire transition matrix for model 3
    transition = transition.diag().expand(num_samples, -1, -1)
    trans_dist = dist.MultivariateNormal(mu_trans, scale_tril=cov_trans).expand((num_samples,))
    obs_dist = dist.MultivariateNormal(mu_obs, scale_tril=cov_obs).expand((num_samples,))
    obs = torch.eye(hidden_dim, obs_dim)
    if init_dist is None:
        z = torch.zeros(num_samples, hidden_dim)
    else:
        z = pyro.sample('z_0', init_dist)

    for i in range(timesteps):
        trans_noise = pyro.sample('trans_noise', trans_dist)
        z = z.unsqueeze(1).bmm(transition).squeeze(1) + trans_noise
        # add observation noise
        obs_noise = pyro.sample('obs_noise', obs_dist)
        y = z @ obs + obs_noise
        ys.append(y)
    data = torch.stack(ys, 1)
    assert data.shape == (num_samples, timesteps, obs_dim)
    return data


def plot(y, h):
    x = np.arange(0, 500, 2)
    plt.plot(x, h[:, 0][::2], label='data')
    plt.plot(x, y[:, 0][::2], '--', alpha=0.4, label='prediction')
    plt.xlabel("Time")
    plt.ylabel("Returns")
    plt.title("Predicted Volatility")
    plt.legend()
    plt.savefig('stoch_volatility.png')


def main(args):
    pyro.enable_validation(True)
    pyro.set_rng_seed(123)
    # generate synthetic data
    with torch.no_grad():
        data = sequential_model()
    # clear the param store after generating data
    pyro.clear_param_store()
    # MAP estimation
    guide = AutoDelta(model)
    svi = SVI(model, guide, Adam({'lr': args.learning_rate}), Trace_ELBO())
    for i in range(args.num_epochs):
        loss = svi.step(data)
        if i % 10 == 0:
            logging.info('epoch {}: {: 4f}'.format(i, loss))
    for k, v in pyro.get_param_store().items():
        print(k, v.detach().cpu().numpy())
    # plot model generated data against observation for one of the stocks
    predictions = sequential_model(init_dist=model(data))
    plot(data[0].cpu().numpy(), predictions[0].detach().cpu().numpy())


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.4.0')
    parser = argparse.ArgumentParser(description="Stochastic volatility")
    parser.add_argument("-n", "--num-epochs", default=200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-2, type=float)
    args = parser.parse_args()
    main(args)
