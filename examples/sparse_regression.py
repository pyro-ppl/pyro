# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import torch
import math

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import init_to_median

from torch.optim import Adam


"""
We demonstrate how to do sparse linear regression using a variant of the
approach described in [1]. This approach is particularly suitable for situations
with many feature dimensions (large P) but not too many datapoints (small N).
In particular we consider a quadratic regressor of the form:

f(X) = constant + sum_i theta_i X_i + sum_{i<j} theta_ij X_i X_j + observation noise

Note that in order to keep the set of identified non-negligible weights theta_i
and theta_ij sparse, the model assumes the weights satisfy a 'strong hierarchy'
condition. See reference [1] for details.

Note that in contrast to [1] we do MAP estimation for the kernel hyperparameters
instead of HMC. This is not expected to be as robust as doing full Bayesian inference,
but in some regimes this works surprisingly well. For the latter HMC approach see
the NumPyro version:

https://github.com/pyro-ppl/numpyro/blob/master/examples/sparse_regression.py

References
[1] The Kernel Interaction Trick: Fast Bayesian Discovery of Pairwise
    Interactions in High Dimensions.
    Raj Agrawal, Jonathan H. Huggins, Brian Trippe, Tamara Broderick
    https://arxiv.org/abs/1905.06501
"""


pyro.enable_validation(True)
torch.set_default_tensor_type('torch.FloatTensor')


def dot(X, Z):
    return torch.mm(X, Z.t())


# The kernel that corresponds to our quadratic regressor.
def kernel(X, Z, eta1, eta2, c):
    eta1sq, eta2sq = eta1.pow(2.0), eta2.pow(2.0)
    k1 = 0.5 * eta2sq * (1.0 + dot(X, Z)).pow(2.0)
    k2 = -0.5 * eta2sq * dot(X.pow(2.0), Z.pow(2.0))
    k3 = (eta1sq - eta2sq) * dot(X, Z)
    k4 = c ** 2 - 0.5 * eta2sq
    return k1 + k2 + k3 + k4


# Most of the model code is concerned with constructing the sparsity inducing prior.
def model(X, Y, hypers, jitter=1.0e-4):
    S, P, N = hypers['expected_sparsity'], X.size(1), X.size(0)

    sigma = pyro.sample("sigma", dist.HalfNormal(hypers['alpha3']))
    phi = sigma * (S / math.sqrt(N)) / (P - S)
    eta1 = pyro.sample("eta1", dist.HalfCauchy(phi))

    msq = pyro.sample("msq", dist.InverseGamma(hypers['alpha1'], hypers['beta1']))
    xisq = pyro.sample("xisq", dist.InverseGamma(hypers['alpha2'], hypers['beta2']))

    eta2 = eta1.pow(2.0) * xisq.sqrt() / msq

    lam = pyro.sample("lambda", dist.HalfCauchy(torch.ones(P, device=X.device)).to_event(1))
    kappa = msq.sqrt() * lam / (msq + (eta1 * lam).pow(2.0)).sqrt()
    kX = kappa * X

    # compute the kernel for the given hyperparameters
    k = kernel(kX, kX, eta1, eta2, hypers['c']) + (sigma ** 2 + jitter) * torch.eye(N, device=X.device)

    # observe the outputs Y
    pyro.sample("Y", dist.MultivariateNormal(torch.zeros(N, device=X.device), covariance_matrix=k),
                obs=Y)


"""
Here we compute the mean and variance of coefficients theta_i (where i = dimension) as well
as for quadratic coefficients theta_ij for a given (in our case MAP) estimate of the kernel
hyperparameters (eta1, xisq, ...).
Compare to theorem 5.1 in reference [1].
"""


@torch.no_grad()
def compute_posterior_stats(X, Y, msq, lam, eta1, xisq, c, sigma, jitter=1.0e-4):
    N, P = X.shape

    # prepare for computation of posterior statistics for singleton weights
    probe = torch.zeros((P, 2, P), dtype=X.dtype, device=X.device)
    probe[:, 0, :] = torch.eye(P, dtype=X.dtype, device=X.device)
    probe[:, 1, :] = -torch.eye(P, dtype=X.dtype, device=X.device)

    eta2 = eta1.pow(2.0) * xisq.sqrt() / msq
    kappa = msq.sqrt() * lam / (msq + (eta1 * lam).pow(2.0)).sqrt()

    kX = kappa * X
    kprobe = kappa * probe
    kprobe = kprobe.reshape(-1, P)

    # compute various kernels
    k_xx = kernel(kX, kX, eta1, eta2, c) + (jitter + sigma ** 2) * torch.eye(N, dtype=X.dtype, device=X.device)
    k_xx_inv = torch.inverse(k_xx)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    # compute mean and variance for singleton weights
    vec = torch.tensor([0.50, -0.50], dtype=X.dtype, device=X.device)
    mu = torch.matmul(k_probeX, torch.matmul(k_xx_inv, Y).unsqueeze(-1)).squeeze(-1).reshape(P, 2)
    mu = (mu * vec).sum(-1)

    var = k_prbprb - torch.matmul(k_probeX, torch.matmul(k_xx_inv, k_probeX.t()))
    var = var.reshape(P, 2, P, 2).diagonal(dim1=-4, dim2=-2)  # 2 2 P
    std = ((var * vec.unsqueeze(-1)).sum(-2) * vec.unsqueeze(-1)).sum(-2).clamp(min=0.0).sqrt()

    # select active dimensions (those that are non-zero with sufficient statistical significance)
    active_dims = (((mu - 4.0 * std) > 0.0) | ((mu + 4.0 * std) < 0.0)).bool()
    active_dims = active_dims.nonzero().squeeze(-1)

    print("Identified the following active dimensions:", active_dims.data.numpy().flatten())
    print("Mean estimate for active singleton weights:\n", mu[active_dims].data.numpy())

    # if there are 0 or 1 active dimensions there are no quadratic weights to be found
    M = len(active_dims)
    if M < 2:
        return active_dims.data.numpy(), []

    # prep for computation of posterior statistics for quadratic weights
    left_dims, right_dims = torch.ones(M, M).triu(1).nonzero().t()
    left_dims, right_dims = active_dims[left_dims], active_dims[right_dims]

    probe = torch.zeros(left_dims.size(0), 4, P, dtype=X.dtype, device=X.device)
    left_dims_expand = left_dims.unsqueeze(-1).expand(left_dims.size(0), P)
    right_dims_expand = right_dims.unsqueeze(-1).expand(right_dims.size(0), P)
    for dim, value in zip(range(4), [1.0, 1.0, -1.0, -1.0]):
        probe[:, dim, :].scatter_(-1, left_dims_expand, value)
    for dim, value in zip(range(4), [1.0, -1.0, 1.0, -1.0]):
        probe[:, dim, :].scatter_(-1, right_dims_expand, value)

    kprobe = kappa * probe
    kprobe = kprobe.reshape(-1, P)
    k_probeX = kernel(kprobe, kX, eta1, eta2, c)
    k_prbprb = kernel(kprobe, kprobe, eta1, eta2, c)

    # compute mean and covariance for a subset of weights theta_ij (namely those with
    # 'active' dimensions i and j)
    vec = torch.tensor([0.25, -0.25, -0.25, 0.25], dtype=X.dtype, device=X.device)
    mu = torch.matmul(k_probeX, torch.matmul(k_xx_inv, Y).unsqueeze(-1)).squeeze(-1).reshape(left_dims.size(0), 4)
    mu = (mu * vec).sum(-1)

    var = k_prbprb - torch.matmul(k_probeX, torch.matmul(k_xx_inv, k_probeX.t()))
    var = var.reshape(left_dims.size(0), 4, left_dims.size(0), 4).diagonal(dim1=-4, dim2=-2)
    std = ((var * vec.unsqueeze(-1)).sum(-2) * vec.unsqueeze(-1)).sum(-2).clamp(min=0.0).sqrt()

    active_quad_dims = (((mu - 4.0 * std) > 0.0) | ((mu + 4.0 * std) < 0.0)) & (mu.abs() > 1.0e-4).bool()
    active_quad_dims = active_quad_dims.nonzero()

    active_quadratic_dims = np.stack([left_dims[active_quad_dims].data.numpy().flatten(),
                                      right_dims[active_quad_dims].data.numpy().flatten()], axis=1)
    active_quadratic_dims = np.split(active_quadratic_dims, active_quadratic_dims.shape[0])
    active_quadratic_dims = [tuple(a.tolist()[0]) for a in active_quadratic_dims]

    return active_dims.data.numpy(), active_quadratic_dims


# Create an artifical dataset with N datapoints and P feature dimensions. Of the P
# dimensions S will have non-zero singleton weights and Q(Q-1)/2 pairs of feature dimensions
# will have non-zero quadratic weights.
def get_data(N=20, P=10, S=2, Q=2, sigma_obs=0.15):
    assert S < P and P > 3 and S > 2 and Q > 1 and Q <= S
    torch.manual_seed(1)

    X = torch.randn(N, P)

    singleton_weights = 2.0 * torch.rand(S) - 1.0
    Y_mean = torch.einsum("ni,i->n", X[:, 0:S], singleton_weights)

    quadratic_weights = []
    expected_quad_dims = []
    for dim1 in range(Q):
        for dim2 in range(Q):
            if dim1 >= dim2:
                continue
            expected_quad_dims.append((dim1, dim2))
            quadratic_weights.append(2.0 * torch.rand(1) - 1.0)
            Y_mean += quadratic_weights[-1] * X[:, dim1] * X[:, dim2]
    quadratic_weights = torch.tensor(quadratic_weights)

    # we standardize the outputs Y
    Y = Y_mean
    Y -= Y.mean()
    Y_std1 = Y.std()
    Y /= Y_std1
    Y += sigma_obs * torch.randn(N)
    Y -= Y.mean()
    Y_std2 = Y.std()
    Y /= Y_std2

    assert X.shape == (N, P)
    assert Y.shape == (N,)

    return X, Y, singleton_weights / (Y_std1 * Y_std2), expected_quad_dims


def init_loc_fn(site):
    value = init_to_median(site, num_samples=50)
    # we also make sure the initial sigma is not too large.
    # (otherwise we run the danger of getting stuck in bad local optima during optimization).
    if site["name"] == "sigma":
        value = 0.1 * value
    return value


def main(args):
    # setup hyperparameters for the model
    hypers = {'expected_sparsity': max(1.0, args.num_dimensions / 10),
              'alpha1': 3.0, 'beta1': 1.0, 'alpha2': 3.0, 'beta2': 1.0, 'alpha3': 1.0,
              'c': 1.0}

    P = args.num_dimensions
    S = args.active_dimensions
    Q = args.quadratic_dimensions

    # generate artificial dataset
    X, Y, expected_thetas, expected_quad_dims = get_data(N=args.num_data, P=P, S=S,
                                                         Q=Q, sigma_obs=args.sigma)

    loss_fn = Trace_ELBO().differentiable_loss

    # We initialize the AutoDelta guide (for MAP estimation) with args.num_trials many
    # initial parameters sampled from the vicinity of the median of the prior distribution
    # and then continue optimizing with the best performing initialization.
    init_losses = []
    for restart in range(args.num_restarts):
        pyro.clear_param_store()
        pyro.set_rng_seed(restart)
        guide = AutoDelta(model, init_loc_fn=init_loc_fn)
        with torch.no_grad():
            init_losses.append(loss_fn(model, guide, X, Y, hypers).item())

    pyro.set_rng_seed(np.argmin(init_losses))
    pyro.clear_param_store()
    guide = AutoDelta(model, init_loc_fn=init_loc_fn)

    # Instead of using pyro.infer.SVI and pyro.optim we instead construct our own PyTorch
    # optimizer and take charge of gradient-based optimization ourselves.
    with poutine.block(), poutine.trace(param_only=True) as param_capture:
        guide(X, Y, hypers)
    params = list([pyro.param(name).unconstrained() for name in param_capture.trace])
    adam = Adam(params, lr=args.lr)

    report_frequency = 50
    print("Beginning MAP optimization...")

    # the optimization loop
    for step in range(args.num_steps):
        loss = loss_fn(model, guide, X, Y, hypers) / args.num_data
        loss.backward()
        adam.step()
        adam.zero_grad()

        # we manually reduce the learning rate according to this schedule
        if step in [100, 300, 700, 900]:
            adam.param_groups[0]['lr'] *= 0.2

        if step % report_frequency == 0 or step == args.num_steps - 1:
            print("[step %04d]  loss: %.5f" % (step, loss))

    print("Expected singleton thetas:\n", expected_thetas.data.numpy())

    # we do the final computation using double precision
    median = guide.median()  # == mode for MAP inference
    active_dims, active_quad_dims = \
        compute_posterior_stats(X.double(), Y.double(), median['msq'].double(),
                                median['lambda'].double(), median['eta1'].double(),
                                median['xisq'].double(), torch.tensor(hypers['c']).double(),
                                median['sigma'].double())

    expected_active_dims = np.arange(S).tolist()

    tp_singletons = len(set(active_dims) & set(expected_active_dims))
    fp_singletons = len(set(active_dims) - set(expected_active_dims))
    fn_singletons = len(set(expected_active_dims) - set(active_dims))
    singleton_stats = (tp_singletons, fp_singletons, fn_singletons)

    tp_quads = len(set(active_quad_dims) & set(expected_quad_dims))
    fp_quads = len(set(active_quad_dims) - set(expected_quad_dims))
    fn_quads = len(set(expected_quad_dims) - set(active_quad_dims))
    quad_stats = (tp_quads, fp_quads, fn_quads)

    # We report how well we did, i.e. did we recover the sparse set of coefficients
    # that we expected for our artificial dataset?
    print("[SUMMARY STATS]")
    print("Singletons (true positive, false positive, false negative): " +
          "(%d, %d, %d)" % singleton_stats)
    print("Quadratic  (true positive, false positive, false negative): " +
          "(%d, %d, %d)" % quad_stats)


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description='Krylov KIT')
    parser.add_argument('--num-data', type=int, default=750)
    parser.add_argument('--num-steps', type=int, default=1000)
    parser.add_argument('--num-dimensions', type=int, default=100)
    parser.add_argument('--num-restarts', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--active-dimensions', type=int, default=10)
    parser.add_argument('--quadratic-dimensions', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.3)
    args = parser.parse_args()

    main(args)
