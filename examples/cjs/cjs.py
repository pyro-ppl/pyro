"""
We show how to implement several variants of the Cormack-Jolly-Seber (CJS)
[3, 4, 5] model used in ecology to analyze animal capture-recapture data.
For a discussion of these models see reference [1].

Throughout we use the European Dipper (Cinclus cinclus) data from reference [2].
This is Norway's national bird.

Compare to the Stan implementations in [6].

References
[1] Kéry, M., & Schaub, M. (2011). Bayesian population analysis using
    WinBUGS: a hierarchical perspective. Academic Press.
[2] Lebreton, J.D., Burnham, K.P., Clobert, J., & Anderson, D.R. (1992).
    Modeling survival and testing biological hypotheses using marked animals:
    a unified approach with case studies. Ecological monographs, 62(1), 67-118.
[3] Cormack, R.M., 1964. Estimates of survival from the sighting of marked animals.
    Biometrika 51, 429–438.
[4] Jolly, G.M., 1965. Explicit estimates from capture-recapture data with both death
    and immigration-stochastic model. Biometrika 52, 225–247.
[5] Seber, G.A.F., 1965. A note on the multiple recapture census. Biometrika 52, 249–259.
[6] https://github.com/stan-dev/example-models/tree/master/BPA/Ch.07
"""

from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam


def model(data):
    N, T = data.shape
    phi = pyro.sample("phi", dist.Uniform(0.0, 1.0))  # survival probability
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    with pyro.plate("dippers", N, dim=-1):
        z = torch.ones(N)
        # we use this mask to eliminate extraneous log probabilities
        # that arise for a given individual bird before its first capture.
        first_capture_mask = torch.zeros(N).byte()
        for t in pyro.markov(range(T)):
            with poutine.mask(mask=first_capture_mask):
                mu_z_t = first_capture_mask.float() * phi * z + (1 - first_capture_mask.float())
                # we use parallel enumeration to exactly sum out
                # the discrete states z_t.
                z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                                infer={"enumerate": "parallel"})
                mu_y_t = rho * z
                pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                            obs=data[:, t])
            first_capture_mask |= data[:, t].byte()


def main(args):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    data_file = os.path.dirname(os.path.abspath(__file__)) + '/dipper_capture_histories.csv'
    data = torch.tensor(np.genfromtxt(data_file, delimiter=',')).float()[:, 1:]
    N, T = data.shape
    print("Loaded dipper capture history for {} individuals collected over {} years.".format(
          N, T))

    # we use a mean field diagonal normal variational distributions (i.e. guide)
    # for the continuous latent variables.
    guide = AutoDiagonalNormal(poutine.block(model, expose=['phi', 'rho']))

    # since we enumerate the discrete random variables,
    # we need to use TraceEnum_ELBO.
    elbo = TraceEnum_ELBO(max_plate_nesting=1, num_particles=20, vectorize_particles=True)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    losses = []

    print("Beginning training with Stochastic Variational Inference.")

    for step in range(args.num_steps):
        loss = svi.step(data)
        losses.append(loss)
        if step % 20 == 0 and step > 0 or step == args.num_steps - 1:
            print("[iter %03d] loss: %.3f" % (step, np.mean(losses[-20:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSJ capture-recapture model for ecological data")
    parser.add_argument("-n", "--num-steps", default=500, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.003, type=float)
    args = parser.parse_args()
    main(args)
