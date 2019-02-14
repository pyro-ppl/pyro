from __future__ import absolute_import, division, print_function

import argparse
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS

"""
This simple example is intended to demonstrate how to use an LKJ prior with
a multivariate distribution.

It generates entirely random, uncorrelated data, and then attempts to fit a correlation matrix
and vector of variances.
"""


def model(y):
    d = y.shape[1]
    N = y.shape[0]
    # Vector of variances for each of the d variables
    theta = pyro.sample("theta", dist.HalfCauchy(torch.full((d,), 1, dtype=y.dtype)))
    # Lower cholesky factor of a correlation matrix
    eta = y.new_full((1,), 1)
    L_omega = pyro.sample("L_omega", dist.CorrLCholeskyLKJPrior(d, eta))
    # Lower cholesky factor of the covariance matrix
    L_Omega = torch.mm(torch.diag(theta.sqrt()), L_omega)

    # Vector of expectations
    mu = y.new_zeros(d)

    with pyro.plate("observations", N) as n:
        obs = pyro.sample("obs", dist.MultivariateNormal(mu, scale_tril=L_Omega), obs=y[n])
    return obs


def main(args):
    y = torch.randn(args.n, args.num_variables).to(dtype=torch.double)
    if args.cuda:
        y = y.cuda()
    nuts_kernel = NUTS(model, jit_compile=args.jit)
    MCMC(nuts_kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps, num_chains=1).run(y)


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.0')
    parser = argparse.ArgumentParser(description="Demonstrate the use of an LKJ Prior")
    parser.add_argument("--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--n", nargs="?", default=500, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument("--num-variables", nargs='?', default=5, type=int)
    parser.add_argument("--warmup-steps", nargs='?', default=100, type=int)
    parser.add_argument("--rng_seed", nargs='?', default=0, type=int)
    parser.add_argument("--jit", action="store_true", default=False,
                        help="use PyTorch jit")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="run this example in GPU")
    args = parser.parse_args()

    pyro.set_rng_seed(args.rng_seed)
    # Enable validation checks
    pyro.enable_validation(True)

    # work around with the error "RuntimeError: received 0 items of ancdata"
    # see https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823
    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.multiprocessing.set_start_method("spawn", force=True)

    main(args)
