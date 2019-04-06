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


"""
Our first and simplest CJS model variant only has two continuous
(scalar) latent random variables: i) the survival probability phi;
and ii) the recapture probability rho. These are treated as fixed
effects with no temporal or individual/group variation.
"""
def model_1(data):
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

"""
In our second model variant there is a survival probability phi_t for 6
of the 7 years of the capture data; each phi_t is treated as a fixed effect.
"""
def model_2(data):
    N, T = data.shape
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    z = torch.ones(N)
    first_capture_mask = torch.zeros(N).byte()
    # we create the plate once, outside of the loop over t
    dippers_plate = pyro.plate("dippers", N, dim=-1)
    for t in pyro.markov(range(T)):
        # note that phi_t needs to be outside the plate, since
        # phi_t is shared across all N birds
        phi_t = pyro.sample("phi_{}".format(t), dist.Uniform(0.0, 1.0)) if t > 0 \
                else 1.0
        with dippers_plate, poutine.mask(mask=first_capture_mask):
            mu_z_t = first_capture_mask.float() * phi_t * z + (1 - first_capture_mask.float())
            # we use parallel enumeration to exactly sum out
            # the discrete states z_t.
            z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                            infer={"enumerate": "parallel"})
            mu_y_t = rho * z
            pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                            obs=data[:, t])
        first_capture_mask |= data[:, t].byte()


"""
In our third model variant there is a survival probability phi_t for 6
of the 7 years of the capture data (just like in model_2), but here
each phi_t is treated as a random effect.
"""
def model_3(data):
    def logit(p):
        return torch.log(p / (1.0 - p))
    N, T = data.shape
    phi_mean = pyro.sample("phi_mean", dist.Uniform(0.0, 1.0))  # mean survival probability
    phi_logit_mean = logit(phi_mean)
    # controls temporal variability of survival probability
    phi_sigma = pyro.sample("phi_sigma", dist.Uniform(0.0, 10.0))
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    z = torch.ones(N)
    first_capture_mask = torch.zeros(N).byte()
    # we create the plate once, outside of the loop over t
    dippers_plate = pyro.plate("dippers", N, dim=-1)
    for t in pyro.markov(range(T)):
        phi_logit_t = pyro.sample("phi_logit_{}".format(t),
                                  dist.Normal(phi_logit_mean, phi_sigma)) if t > 0 \
                      else torch.tensor(0.0)
        phi_t = torch.sigmoid(phi_logit_t)
        with dippers_plate, poutine.mask(mask=first_capture_mask):
            mu_z_t = first_capture_mask.float() * phi_t * z + (1 - first_capture_mask.float())
            # we use parallel enumeration to exactly sum out
            # the discrete states z_t.
            z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                            infer={"enumerate": "parallel"})
            mu_y_t = rho * z
            pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                        obs=data[:, t])
        first_capture_mask |= data[:, t].byte()




models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)

    data_file = os.path.dirname(os.path.abspath(__file__)) + '/dipper_capture_histories.csv'
    data = torch.tensor(np.genfromtxt(data_file, delimiter=',')).float()[:, 1:]
    N, T = data.shape
    print("Loaded dipper capture history for {} individuals collected over {} years.".format(
          N, T))

    model = models[args.model]
    # we use a mean field diagonal normal variational distributions (i.e. guide)
    # for the continuous latent variables.
    expose_fn = lambda msg: msg["name"][0:3] in ['phi', 'rho']
    guide = AutoDiagonalNormal(poutine.block(model, expose_fn=expose_fn))

    # since we enumerate the discrete random variables,
    # we need to use TraceEnum_ELBO.
    elbo = TraceEnum_ELBO(max_plate_nesting=1, num_particles=20, vectorize_particles=True)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    losses = []

    print("Beginning training of model_{} with Stochastic Variational Inference.".format(args.model))

    for step in range(args.num_steps):
        loss = svi.step(data)
        losses.append(loss)
        if step % 20 == 0 and step > 0 or step == args.num_steps - 1:
            print("[iter %03d] loss: %.3f" % (step, np.mean(losses[-20:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CSJ capture-recapture model for ecological data")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-n", "--num-steps", default=300, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.002, type=float)
    args = parser.parse_args()
    main(args)
