"""
This example demonstrates the functionality of `pyro.contrib.minipyro`,
which is a minimal implementation of the Pyro Probabilistic Programming
Language that was created for didactic purposes.
"""

from __future__ import absolute_import, division, print_function

import argparse

import torch

# We use the pyro.generic interface to support dynamic choice of backend.
from pyro.generic import pyro_backend
from pyro.generic import distributions as dist
from pyro.generic import infer, optim, pyro


def main(args):
    # Define a basic model with a single Normal latent random variable `loc`
    # and a batch of Normally distributed observations.
    def model(data):
        loc = pyro.sample("loc", dist.Normal(0., 1.))
        with pyro.plate("data", len(data), dim=-1):
            pyro.sample("obs", dist.Normal(loc, 1.), obs=data)

    # Define a guide (i.e. variational distribution) with a Normal
    # distribution over the latent random variable `loc`.
    def guide(data):
        guide_loc = pyro.param("guide_loc", torch.tensor(0.))
        guide_scale = pyro.param("guide_scale_log", torch.tensor(0.)).exp()
        pyro.sample("loc", dist.Normal(guide_loc, guide_scale))

    # Generate some data.
    torch.manual_seed(0)
    data = torch.randn(100) + 3.0

    # Because the API in minipyro matches that of Pyro proper,
    # training code works with generic Pyro implementations.
    with pyro_backend(args.backend):
        # Construct an SVI object so we can do variational inference on our
        # model/guide pair.
        elbo = infer.Trace_ELBO()
        adam = optim.Adam({"lr": args.learning_rate})
        svi = infer.SVI(model, guide, adam, elbo)

        # Basic training loop
        pyro.get_param_store().clear()
        for step in range(args.num_steps):
            loss = svi.step(data)
            if step % 100 == 0:
                print("step {} loss = {}".format(step, loss))

        # Report the final values of the variational parameters
        # in the guide after training.
        for name, value in pyro.get_param_store().items():
            print("{} = {}".format(name, value.detach().cpu().numpy()))

        # For this simple (conjugate) model we know the exact posterior. In
        # particular we know that the variational distribution should be
        # centered near 3.0. So let's check this explicitly.
        assert (pyro.param("guide_loc") - 3.0).abs() < 0.1


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.1')
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-b", "--backend", default="minipyro")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    args = parser.parse_args()
    main(args)
