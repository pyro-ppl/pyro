# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Distributed training via Pytorch Lightning.
#
# This tutorial demonstrates how to distribute SVI training across multiple
# machines (or multiple GPUs on one or more machines) using the PyTorch Lightning
# library. PyTorch Lightning enables data-parallel training by aggregating stochastic
# gradients at each step of training. We focus on integration between PyTorch Lightning and Pyro.
# For further details on distributed computing with PyTorch Lightning, see
#   https://lightning.ai/docs/pytorch/latest
#
# This assumes you have installed pytorch lightning, e.g. via
#   pip install pyro[lightning]

import argparse

import pytorch_lightning as pl
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule


# We define a model as usual, with no reference to Pytorch Lightning.
# This model is data parallel and supports subsampling.
class Model(PyroModule):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, covariates, data=None):
        coeff = pyro.sample("coeff", dist.Normal(0, 1))
        bias = pyro.sample("bias", dist.Normal(0, 1))
        scale = pyro.sample("scale", dist.LogNormal(0, 1))

        # Since we'll use a distributed dataloader during training, we need to
        # manually pass minibatches of (covariates,data) that are smaller than
        # the full self.size. In particular we cannot rely on pyro.plate to
        # automatically subsample, since that would lead to all workers drawing
        # identical subsamples.
        with pyro.plate("data", self.size, len(covariates)):
            loc = bias + coeff * covariates
            return pyro.sample("obs", dist.Normal(loc, scale), obs=data)


# We define an ELBO loss, a PyTorch optimizer, and a training step in our PyroLightningModule
# Note that we are using PyTorch optimizer and we are not using Pyro's SVI machinery
class PyroLightningModule(pl.LightningModule):
    def __init__(self, model, guide, lr):
        super().__init__()
        self.pyro_model = model
        self.pyro_guide = guide
        # Create an ELBO loss function
        self.loss_fn = Trace_ELBO().differentiable_loss
        self.lr = lr

    def training_step(self, batch, batch_idx):
        """Training step for Pyro training."""
        loss = self.loss_fn(self.pyro_model, self.pyro_guide, *batch)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        return torch.optim.Adam(self.pyro_guide.parameters(), lr=self.lr)


def main(args):
    # Create a model, synthetic data, a guide, and a lightning module.
    pyro.set_rng_seed(args.seed)
    model = Model(args.size)
    covariates = torch.randn(args.size)
    data = model(covariates)
    guide = AutoNormal(model)
    training_plan = PyroLightningModule(model, guide, args.learning_rate)

    # Create a dataloader.
    dataset = torch.utils.data.TensorDataset(covariates, data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    # All relevant parameters need to be initialized before `configure_optimizer` is called.
    # Since we used AutoNormal guide our parameters have not be initialized yet.
    # Warm up the guide by running one mini-batch through it.
    mini_batch = dataset[: args.batch_size]
    guide(*mini_batch)

    # Run stochastic variational inference using PyTorch Lightning Trainer.
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(training_plan, train_dataloaders=dataloader)


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.4")
    parser = argparse.ArgumentParser(
        description="Distributed training via PyTorch Lightning"
    )
    parser.add_argument("--size", default=1000000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--lightning", action="store_true", default=True)
    parser.add_argument("--seed", default=20200723, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
