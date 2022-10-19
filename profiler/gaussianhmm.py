# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from tqdm.auto import tqdm

import pyro.distributions as dist


def random_mvn(batch_shape, dim):
    rank = dim + dim
    loc = torch.randn(batch_shape + (dim,), requires_grad=True)
    cov = torch.randn(batch_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    scale_tril = torch.linalg.cholesky(cov)
    scale_tril.requires_grad_()
    return dist.MultivariateNormal(loc, scale_tril=scale_tril)


def find_parameters(params: list, *xs) -> None:
    for x in xs:
        if isinstance(x, torch.Tensor):
            if x.requires_grad:
                params.append(x)
        elif isinstance(x, dist.Distribution):
            find_parameters(params, *x.__dict__.values())


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    hidden_dim = args.hidden_dim
    obs_dim = args.obs_dim
    duration = args.duration
    batch_shape = (args.batch_size,)

    # Initialize parts.
    init_dist = random_mvn(batch_shape, hidden_dim)
    trans_dist = random_mvn(batch_shape + (duration,), hidden_dim)
    obs_dist = random_mvn(batch_shape + (1,), obs_dim)
    trans_mat = (
        torch.randn(batch_shape + (duration, hidden_dim, hidden_dim))
        .mul_(0.1)
        .requires_grad_()
    )
    obs_mat = torch.randn(batch_shape + (1, hidden_dim, obs_dim), requires_grad=True)

    # Collect parameters.
    params = []
    find_parameters(params, init_dist, trans_dist, obs_dist, trans_mat, obs_mat)
    assert params

    # Build a distribution.
    d = dist.GaussianHMM(
        init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=duration
    )

    # Time forward + backward.
    for step in tqdm(range(args.num_steps)):
        x = d.rsample()
        grads = torch.autograd.grad(
            x.sum(), params, allow_unused=True, retain_graph=True
        )
        assert not all(g is None for g in grads)
        del x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaussianHMM profiler")
    parser.add_argument("--hidden-dim", type=int, default=10)
    parser.add_argument("--obs-dim", type=int, default=20)
    parser.add_argument("--duration", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)
