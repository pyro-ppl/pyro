# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import torch
from tqdm.auto import tqdm

import pyro.distributions as dist


def random_mvn(batch_shape, dim, requires_grad=False):
    rank = dim + dim
    loc = torch.randn(batch_shape + (dim,), requires_grad=requires_grad)
    cov = torch.randn(batch_shape + (dim, rank))
    cov = cov.matmul(cov.transpose(-1, -2))
    scale_tril = torch.linalg.cholesky(cov)
    scale_tril.requires_grad_(requires_grad)
    return dist.MultivariateNormal(loc, scale_tril=scale_tril)


def main(args):
    if args.cuda:
        torch.set_default_device("cuda")

    hidden_dim = args.hidden_dim
    obs_dim = args.obs_dim
    duration = args.duration
    batch_shape = (args.batch_size,)

    # Initialize parts.
    init_dist = random_mvn(batch_shape, hidden_dim, requires_grad=args.grad)
    trans_dist = random_mvn(
        batch_shape + (duration,), hidden_dim, requires_grad=args.grad
    )
    obs_dist = random_mvn(batch_shape + (1,), obs_dim, requires_grad=args.grad)
    trans_mat = 0.1 * torch.randn(batch_shape + (duration, hidden_dim, hidden_dim))
    obs_mat = torch.randn(batch_shape + (1, hidden_dim, obs_dim))

    if args.grad:
        # Collect parameters.
        params = [
            init_dist.loc,
            init_dist.scale_tril,
            trans_dist.loc,
            trans_dist.scale_tril,
            obs_dist.loc,
            obs_dist.scale_tril,
            trans_mat.requires_grad_(),
            obs_mat.requires_grad_(),
        ]

    # Build a distribution.
    d = dist.GaussianHMM(
        init_dist, trans_mat, trans_dist, obs_mat, obs_dist, duration=duration
    )

    for step in tqdm(range(args.num_steps)):
        if not args.grad:
            # Time forward only.
            d.sample()
            continue

        # Time forward + backward.
        x = d.rsample()
        grads = torch.autograd.grad(
            x.sum(), params, allow_unused=True, retain_graph=True
        )
        assert not all(g is None for g in grads)
        del x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GaussianHMM profiler")
    parser.add_argument("--hidden-dim", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument("--duration", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("-n", "--num-steps", type=int, default=100)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--grad", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
