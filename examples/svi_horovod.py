# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# Distributed training via Horovod.
#
# This tutorial demonstrates how to distribute SVI training across multiple
# machines (or multiple GPUs on one or more machines) using the Horovod
# library. Horovod enables data-parallel training by aggregating stochastic
# gradients at each step of training. Horovod is not intended for model
# parallelism. We focus on integration between Horovod and Pyro. For further
# details on distributed computing with Horovod, see
#   https://horovod.readthedocs.io/en/stable
#
# This assumes you have installed horovod, e.g. via
#   pip install pyro[horovod]
# For detailed instructions see
#   https://horovod.readthedocs.io/en/stable/install.html
# On my mac laptop I was able to install horovod with
#   CFLAGS=-mmacosx-version-min=10.15 HOROVOD_WITH_PYTORCH=1 \
#   pip install --no-cache-dir 'horovod[pytorch]'
#
# Finally, you'll need to run this script via horovodrun, e.g.
#   horovodrun -np 2 python svi_horovod.py
# For details on running Horovod see
#   https://github.com/horovod/horovod/blob/master/docs/running.rst

import argparse

import torch
import torch.multiprocessing as mp

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule
from pyro.optim import Adam, HorovodOptimizer


# We define a model as usual, with no reference to Horovod.
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
            return pyro.sample("obs", dist.Normal(loc, scale),
                               obs=data)


# The following is a standard training loop. To emphasize the Horovod-specific
# parts, we've guarded them by `if args.horovod:`.
def main(args):
    # Create a model, synthetic data, and a guide.
    pyro.set_rng_seed(args.seed)
    model = Model(args.size)
    covariates = torch.randn(args.size)
    data = model(covariates)
    guide = AutoNormal(model)

    if args.horovod:
        # Initialize Horovod and set PyTorch globals.
        import horovod.torch as hvd
        hvd.init()
        torch.set_num_threads(1)
        if args.cuda:
            torch.cuda.set_device(hvd.local_rank())
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    device = torch.tensor(0).device

    if args.horovod:
        # Initialize parameters and broadcast to all workers.
        guide(covariates[:1], data[:1])  # Initializes model and guide.
        hvd.broadcast_parameters(guide.state_dict(), root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Create an ELBO loss and a Pyro optimizer.
    elbo = Trace_ELBO()
    optim = Adam({"lr": args.learning_rate})

    if args.horovod:
        # Wrap the basic optimizer in a distributed optimizer.
        optim = HorovodOptimizer(optim)

    # Create a dataloader.
    dataset = torch.utils.data.TensorDataset(covariates, data)
    if args.horovod:
        # Horovod requires a distributed sampler.
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, hvd.size(), hvd.rank())
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    config = {"batch_size": args.batch_size, "sampler": sampler}
    if args.cuda:
        config["num_workers"] = 1
        config["pin_memory"] = True
        # Try to use forkserver to spawn workers instead of fork.
        if (hasattr(mp, "_supports_context") and mp._supports_context and
                "forkserver" in mp.get_all_start_methods()):
            config["multiprocessing_context"] = "forkserver"
    dataloader = torch.utils.data.DataLoader(dataset, **config)

    # Run stochastic variational inference.
    svi = SVI(model, guide, optim, elbo)
    for epoch in range(args.num_epochs):
        if args.horovod:
            # Set rng seeds on distributed samplers. This is required.
            sampler.set_epoch(epoch)

        for step, (covariates_batch, data_batch) in enumerate(dataloader):
            loss = svi.step(covariates_batch.to(device), data_batch.to(device))

            if args.horovod:
                # Optionally average loss metric across workers.
                # You can do this with arbitrary torch.Tensors.
                loss = torch.tensor(loss)
                loss = hvd.allreduce(loss, "loss")
                loss = loss.item()

                # Print only on the rank=0 worker.
                if step % 100 == 0 and hvd.rank() == 0:
                    print("epoch {} step {} loss = {:0.4g}".format(epoch, step, loss))
            else:
                if step % 100 == 0:
                    print("epoch {} step {} loss = {:0.4g}".format(epoch, step, loss))

    if args.horovod:
        # After we're done with the distributed parts of the program,
        # we can shutdown all but the rank=0 worker.
        hvd.shutdown()
        if hvd.rank() != 0:
            return

    if args.outfile:
        print("saving to {}".format(args.outfile))
        torch.save({"model": model, "guide": guide}, args.outfile)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(description="Distributed training via Horovod")
    parser.add_argument("-o", "--outfile")
    parser.add_argument("-s", "--size", default=1000000, type=int)
    parser.add_argument("-b", "--batch-size", default=100, type=int)
    parser.add_argument("-n", "--num-epochs", default=10, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--horovod", action="store_true", default=True)
    parser.add_argument("--no-horovod", action="store_false", dest="horovod")
    parser.add_argument("--seed", default=20200723, type=int)
    args = parser.parse_args()
    main(args)
