from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.distributions import constraints, Bernoulli

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam


def model(data, phi=0.7, theta=0.8):
    N, T = data.shape
    fmask = torch.zeros(N).byte()
    print("data\n", data.data.numpy())

    with pyro.plate("animals", N, dim=-1):
        z = torch.ones(N)
        for t in pyro.markov(range(T)):
            with pyro.poutine.mask(mask=fmask):
                print("\nfmask[%d] " % t, fmask.data.numpy())
                mu_z_t = fmask.float() * phi * z + (1 - fmask.float())
                print("mu_z_%d" % t, mu_z_t.data.numpy())
                z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                                infer={"enumerate": "parallel"})
                print('z_%d' % t, z.data.numpy())
                mu_y_t = theta * z
                print("mu_y_%d" % t, mu_y_t.data.numpy())
                pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                            obs=data[:, t])
                print("y_%d" % t, data[:, t].data.numpy())
                fmask |= data[:, t].byte()


def guide(data, phi=0.7, theta=0.8):
    pass


def main(args):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    N, T = 8, 6
    data = Bernoulli(0.55).sample(sample_shape=(N, T)).float()
    data[0:3, 0] = torch.zeros(3)
    model(data)
    #guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("probs_")))

    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    optim = Adam({'lr': 0.01})
    svi = SVI(model, guide, optim, elbo)

    print("\ncommencing training...")
    for step in range(args.num_steps):
        loss = svi.step(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSJ capture-recapture model for ecological data")
    parser.add_argument("-n", "--num-steps", default=1, type=int)
    args = parser.parse_args()
    main(args)
