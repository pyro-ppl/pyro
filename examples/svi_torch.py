# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Using vanilla PyTorch to perform optimization in SVI.
#
# This tutorial demonstrates how to use standard PyTorch optimizers, dataloaders and training loops
# to perform optimization in SVI. This is useful when you want to use custom optimizers,
# learning rate schedules, dataloaders, or other advanced training techniques,
# or just to simplify integration with other elements of the PyTorch ecosystem.

import argparse
from typing import Callable

import torch

import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule


# We define a model as usual. This model is data parallel and supports subsampling.
class Model(PyroModule):
    def __init__(self, size):
        super().__init__()
        self.size = size
        # We register a buffer for a constant scalar tensor to represent zero.
        # This is useful for making priors that do not depend on inputs
        # or learnable parameters compatible with the Module.to() method
        # for setting the device or dtype of a module and its parameters.
        self.register_buffer("zero", torch.tensor(0.0))

    def forward(self, covariates, data=None):
        # Sample parameters from priors that make use of the zero buffer trick
        coeff = pyro.sample("coeff", dist.Normal(self.zero, 1))
        bias = pyro.sample("bias", dist.Normal(self.zero, 1))
        scale = pyro.sample("scale", dist.LogNormal(self.zero, 1))

        # Since we'll use a PyTorch dataloader during training, we need to
        # manually pass minibatches of (covariates,data) that are smaller than
        # the full self.size, rather than relying on pyro.plate to automatically subsample.
        with pyro.plate("data", self.size, len(covariates)):
            loc = bias + coeff * covariates
            return pyro.sample("obs", dist.Normal(loc, scale), obs=data)


def main(args):
    # Make PyroModule parameters local (like ordinary torch.nn.Parameters),
    # rather than shared by name through Pyro's global parameter store.
    # This is highly recommended whenever models can be written without pyro.param().
    pyro.settings.set(module_local_params=True)

    # set seed for reproducibility
    pyro.set_rng_seed(args.seed)

    # Create a synthetic dataset from a randomly initialized model.
    with torch.no_grad():
        covariates = torch.randn(args.size)
        data = Model(args.size)(covariates)
        covariates = covariates.to(device=torch.device("cuda" if args.cuda else "cpu"))
        data = data.to(device=torch.device("cuda" if args.cuda else "cpu"))

    # Create a model and a guide, both as (Pyro)Modules.
    model: torch.nn.Module = Model(args.size)
    guide: torch.nn.Module = AutoNormal(model)

    # Create a loss function as a Module that includes model and guide parameters.
    # All Pyro ELBO estimators can be __call__()ed with a model and guide pair as arguments
    # to return a loss function Module that takes the same arguments as the model and guide
    # and exposes all of their torch.nn.Parameters and pyro.nn.PyroParam parameters.
    elbo: Callable[[torch.nn.Module, torch.nn.Module], torch.nn.Module] = Trace_ELBO()
    loss_fn: torch.nn.Module = elbo(model, guide)
    loss_fn.to(device=torch.device("cuda" if args.cuda else "cpu"))

    # Create a dataloader.
    dataset = torch.utils.data.TensorDataset(covariates, data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    # All relevant parameters need to be initialized before an optimizer can be created.
    # Since we used AutoNormal guide our parameters have not be initialized yet.
    # Therefore we initialize the model and guide by running one mini-batch through the loss.
    mini_batch = dataset[: args.batch_size]
    loss_fn(*mini_batch)

    # Create a PyTorch optimizer for the parameters of the model and guide in loss_fn.
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=args.learning_rate)

    # Run stochastic variational inference using PyTorch optimizers from torch.optim
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(*batch)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch} loss = {loss}")


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.9.1")
    parser = argparse.ArgumentParser(
        description="Using vanilla PyTorch to perform optimization in SVI"
    )
    parser.add_argument("--size", default=10000, type=int)
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--seed", default=20200723, type=int)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
