import argparse

import torch
import math
import traceback

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import NUTS, HMC
from pyro.infer.mcmc.api import MCMC
from pyro import sample

pyro.enable_validation(True)
pyro.set_rng_seed(0)
torch.set_default_tensor_type('torch.DoubleTensor')


def dot(X, Z):
    return torch.mm(X, Z.t())


def kernel(X, Z, eta1, eta2, c, jitter=1.0e-4):
    eta1sq, eta2sq = eta1.pow(2.0), eta2.pow(2.0)
    k1 = 0.5 * eta2sq * (1.0 + dot(X, Z)).pow(2.0)
    k2 = -0.5 * eta2sq * dot(X.pow(2.0), Z.pow(2.0))
    k3 = (eta1sq - eta2sq) * dot(X, Z)
    k4 = c ** 2 - 0.5 * eta2sq
    #if X.shape == Z.shape:
    k4 = k4 + jitter * torch.eye(X.size(0), device=X.device)
    return k1 + k2 + k3 + k4


def model(X, Y, hypers):
    S, P, N = hypers['expected_sparsity'], X.size(1), X.size(0)

    sigma = sample("sigma", dist.HalfNormal(hypers['alpha3']))
    phi = sigma * (S / math.sqrt(N)) / (P - S)
    eta1 = sample("eta1", dist.HalfCauchy(phi))

    msq = sample("msq", dist.InverseGamma(hypers['alpha1'], hypers['beta1']))
    xisq = sample("xisq", dist.InverseGamma(hypers['alpha2'], hypers['beta2']))

    eta2 = eta1.pow(2.0) * xisq.sqrt() / msq

    lam = sample("lambda", dist.HalfCauchy(torch.ones(P, device=X.device)))
    kappa = msq.sqrt() * lam / (msq + (eta1 * lam).pow(2.0)).sqrt()

    var_obs = sample("var_obs", dist.InverseGamma(hypers['alpha_obs'], hypers['beta_obs']))

    kX = kappa * X
    k = kernel(kX, kX, eta1, eta2, hypers['c']) + var_obs * torch.eye(N, device=X.device)
    assert k.shape == (N, N)

    sample("Y", dist.MultivariateNormal(loc=torch.zeros(N, device=X.device), covariance_matrix=k), obs=Y)


def get_data(N=20, S=2, P=10, sigma_obs=0.05):
    assert S < P and P > 1 and S > 0
    torch.manual_seed(0)

    X = torch.randn(N, P)
    W = 0.5 + 2.5 * torch.rand(S)
    Y = torch.sum(X[:, 0:S] * W, dim=-1) + X[:, 0] * X[:, 1] + sigma_obs * torch.randn(N)
    Y -= Y.mean()
    Y_std = Y.std()

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y / Y_std, W / Y_std, 1.0 / Y_std


def main(args):
    hypers = {'expected_sparsity': max(1.0, args.num_dimensions / 10),
              'alpha1': 3.0, 'beta1': 1.0,
              'alpha2': 3.0, 'beta2': 1.0,
              'alpha3': 1.0, 'c': 1.0,
              'alpha_obs': 3.0, 'beta_obs': 1.0}

    X, Y, expected_thetas, expected_pairwise = get_data(N=args.num_data, P=args.num_dimensions,
                                                        S=args.active_dimensions)

    try:
        #nuts_kernel = HMC(model, jit_compile=args.jit, step_size=0.05, target_accept_prob=0.95)
        nuts_kernel = NUTS(model, jit_compile=args.jit, step_size=0.05, target_accept_prob=0.95, max_tree_depth=6)
        mcmc = MCMC(nuts_kernel,
                    num_samples=args.num_samples,
                    warmup_steps=args.warmup_steps,
                    num_chains=args.num_chains)
        mcmc.run(X, Y, hypers)
        mcmc.summary(prob=0.5)
    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.4')
    parser = argparse.ArgumentParser(description='KIT')
    parser.add_argument('--num-data', type=int, default=50)
    parser.add_argument('--num-dimensions', type=int, default=6)
    parser.add_argument('--active-dimensions', type=int, default=2)
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument('--num-chains', type=int, default=1)
    parser.add_argument('--warmup-steps', type=int, default=200)
    parser.add_argument('--jit', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
