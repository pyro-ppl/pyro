# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
We show how to implement several variants of the Cormack-Jolly-Seber (CJS)
[4, 5, 6] model used in ecology to analyze animal capture-recapture data.
For a discussion of these models see reference [1].

We make use of two datasets:
-- the European Dipper (Cinclus cinclus) data from reference [2]
   (this is Norway's national bird).
-- the meadow voles data from reference [3].

Compare to the Stan implementations in [7].

References
[1] Kery, M., & Schaub, M. (2011). Bayesian population analysis using
    WinBUGS: a hierarchical perspective. Academic Press.
[2] Lebreton, J.D., Burnham, K.P., Clobert, J., & Anderson, D.R. (1992).
    Modeling survival and testing biological hypotheses using marked animals:
    a unified approach with case studies. Ecological monographs, 62(1), 67-118.
[3] Nichols, Pollock, Hines (1984) The use of a robust capture-recapture design
    in small mammal population studies: A field example with Microtus pennsylvanicus.
    Acta Theriologica 29:357-365.
[4] Cormack, R.M., 1964. Estimates of survival from the sighting of marked animals.
    Biometrika 51, 429-438.
[5] Jolly, G.M., 1965. Explicit estimates from capture-recapture data with both death
    and immigration-stochastic model. Biometrika 52, 225-247.
[6] Seber, G.A.F., 1965. A note on the multiple recapture census. Biometrika 52, 249-259.
[7] https://github.com/stan-dev/example-models/tree/master/BPA/Ch.07
"""

import argparse
import os

import numpy as np
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, TraceEnum_ELBO, TraceTMC_ELBO
from pyro.optim import Adam


"""
Our first and simplest CJS model variant only has two continuous
(scalar) latent random variables: i) the survival probability phi;
and ii) the recapture probability rho. These are treated as fixed
effects with no temporal or individual/group variation.
"""


def model_1(capture_history, sex):
    N, T = capture_history.shape
    phi = pyro.sample("phi", dist.Uniform(0.0, 1.0))  # survival probability
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    with pyro.plate("animals", N, dim=-1):
        z = torch.ones(N)
        # we use this mask to eliminate extraneous log probabilities
        # that arise for a given individual before its first capture.
        first_capture_mask = torch.zeros(N).bool()
        for t in pyro.markov(range(T)):
            with poutine.mask(mask=first_capture_mask):
                mu_z_t = first_capture_mask.float() * phi * z + (1 - first_capture_mask.float())
                # we use parallel enumeration to exactly sum out
                # the discrete states z_t.
                z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                                infer={"enumerate": "parallel"})
                mu_y_t = rho * z
                pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                            obs=capture_history[:, t])
            first_capture_mask |= capture_history[:, t].bool()


"""
In our second model variant there is a time-varying survival probability phi_t for
T-1 of the T time periods of the capture data; each phi_t is treated as a fixed effect.
"""


def model_2(capture_history, sex):
    N, T = capture_history.shape
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    z = torch.ones(N)
    first_capture_mask = torch.zeros(N).bool()
    # we create the plate once, outside of the loop over t
    animals_plate = pyro.plate("animals", N, dim=-1)
    for t in pyro.markov(range(T)):
        # note that phi_t needs to be outside the plate, since
        # phi_t is shared across all N individuals
        phi_t = pyro.sample("phi_{}".format(t), dist.Uniform(0.0, 1.0)) if t > 0 \
                else 1.0
        with animals_plate, poutine.mask(mask=first_capture_mask):
            mu_z_t = first_capture_mask.float() * phi_t * z + (1 - first_capture_mask.float())
            # we use parallel enumeration to exactly sum out
            # the discrete states z_t.
            z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                            infer={"enumerate": "parallel"})
            mu_y_t = rho * z
            pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                        obs=capture_history[:, t])
        first_capture_mask |= capture_history[:, t].bool()


"""
In our third model variant there is a survival probability phi_t for T-1
of the T time periods of the capture data (just like in model_2), but here
each phi_t is treated as a random effect.
"""


def model_3(capture_history, sex):
    def logit(p):
        return torch.log(p) - torch.log1p(-p)
    N, T = capture_history.shape
    phi_mean = pyro.sample("phi_mean", dist.Uniform(0.0, 1.0))  # mean survival probability
    phi_logit_mean = logit(phi_mean)
    # controls temporal variability of survival probability
    phi_sigma = pyro.sample("phi_sigma", dist.Uniform(0.0, 10.0))
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    z = torch.ones(N)
    first_capture_mask = torch.zeros(N).bool()
    # we create the plate once, outside of the loop over t
    animals_plate = pyro.plate("animals", N, dim=-1)
    for t in pyro.markov(range(T)):
        phi_logit_t = pyro.sample("phi_logit_{}".format(t),
                                  dist.Normal(phi_logit_mean, phi_sigma)) if t > 0 \
                      else torch.tensor(0.0)
        phi_t = torch.sigmoid(phi_logit_t)
        with animals_plate, poutine.mask(mask=first_capture_mask):
            mu_z_t = first_capture_mask.float() * phi_t * z + (1 - first_capture_mask.float())
            # we use parallel enumeration to exactly sum out
            # the discrete states z_t.
            z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                            infer={"enumerate": "parallel"})
            mu_y_t = rho * z
            pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                        obs=capture_history[:, t])
        first_capture_mask |= capture_history[:, t].bool()


"""
In our fourth model variant we include group-level fixed effects
for sex (male, female).
"""


def model_4(capture_history, sex):
    N, T = capture_history.shape
    # survival probabilities for males/females
    phi_male = pyro.sample("phi_male", dist.Uniform(0.0, 1.0))
    phi_female = pyro.sample("phi_female", dist.Uniform(0.0, 1.0))
    # we construct a N-dimensional vector that contains the appropriate
    # phi for each individual given its sex (female = 0, male = 1)
    phi = sex * phi_male + (1.0 - sex) * phi_female
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    with pyro.plate("animals", N, dim=-1):
        z = torch.ones(N)
        # we use this mask to eliminate extraneous log probabilities
        # that arise for a given individual before its first capture.
        first_capture_mask = torch.zeros(N).bool()
        for t in pyro.markov(range(T)):
            with poutine.mask(mask=first_capture_mask):
                mu_z_t = first_capture_mask.float() * phi * z + (1 - first_capture_mask.float())
                # we use parallel enumeration to exactly sum out
                # the discrete states z_t.
                z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                                infer={"enumerate": "parallel"})
                mu_y_t = rho * z
                pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                            obs=capture_history[:, t])
            first_capture_mask |= capture_history[:, t].bool()


"""
In our final model variant we include both fixed group effects and fixed
time effects for the survival probability phi:

logit(phi_t) = beta_group + gamma_t

We need to take care that the model is not overparameterized; to do this
we effectively let a single scalar beta encode the difference in male
and female survival probabilities.
"""


def model_5(capture_history, sex):
    N, T = capture_history.shape

    # phi_beta controls the survival probability differential
    # for males versus females (in logit space)
    phi_beta = pyro.sample("phi_beta", dist.Normal(0.0, 10.0))
    phi_beta = sex * phi_beta
    rho = pyro.sample("rho", dist.Uniform(0.0, 1.0))  # recapture probability

    z = torch.ones(N)
    first_capture_mask = torch.zeros(N).bool()
    # we create the plate once, outside of the loop over t
    animals_plate = pyro.plate("animals", N, dim=-1)
    for t in pyro.markov(range(T)):
        phi_gamma_t = pyro.sample("phi_gamma_{}".format(t), dist.Normal(0.0, 10.0)) if t > 0 \
                      else 0.0
        phi_t = torch.sigmoid(phi_beta + phi_gamma_t)
        with animals_plate, poutine.mask(mask=first_capture_mask):
            mu_z_t = first_capture_mask.float() * phi_t * z + (1 - first_capture_mask.float())
            # we use parallel enumeration to exactly sum out
            # the discrete states z_t.
            z = pyro.sample("z_{}".format(t), dist.Bernoulli(mu_z_t),
                            infer={"enumerate": "parallel"})
            mu_y_t = rho * z
            pyro.sample("y_{}".format(t), dist.Bernoulli(mu_y_t),
                        obs=capture_history[:, t])
        first_capture_mask |= capture_history[:, t].bool()


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}


def main(args):
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    # load data
    if args.dataset == "dipper":
        capture_history_file = os.path.dirname(os.path.abspath(__file__)) + '/dipper_capture_history.csv'
    elif args.dataset == "vole":
        capture_history_file = os.path.dirname(os.path.abspath(__file__)) + '/meadow_voles_capture_history.csv'
    else:
        raise ValueError("Available datasets are \'dipper\' and \'vole\'.")

    capture_history = torch.tensor(np.genfromtxt(capture_history_file, delimiter=',')).float()[:, 1:]
    N, T = capture_history.shape
    print("Loaded {} capture history for {} individuals collected over {} time periods.".format(
          args.dataset, N, T))

    if args.dataset == "dipper" and args.model in ["4", "5"]:
        sex_file = os.path.dirname(os.path.abspath(__file__)) + '/dipper_sex.csv'
        sex = torch.tensor(np.genfromtxt(sex_file, delimiter=',')).float()[:, 1]
        print("Loaded dipper sex data.")
    elif args.dataset == "vole" and args.model in ["4", "5"]:
        raise ValueError("Cannot run model_{} on meadow voles data, since we lack sex "
                         "information for these animals.".format(args.model))
    else:
        sex = None

    model = models[args.model]

    # we use poutine.block to only expose the continuous latent variables
    # in the models to AutoDiagonalNormal (all of which begin with 'phi'
    # or 'rho')
    def expose_fn(msg):
        return msg["name"][0:3] in ['phi', 'rho']

    # we use a mean field diagonal normal variational distributions (i.e. guide)
    # for the continuous latent variables.
    guide = AutoDiagonalNormal(poutine.block(model, expose_fn=expose_fn))

    # since we enumerate the discrete random variables,
    # we need to use TraceEnum_ELBO or TraceTMC_ELBO.
    optim = Adam({'lr': args.learning_rate})
    if args.tmc:
        elbo = TraceTMC_ELBO(max_plate_nesting=1)
        tmc_model = poutine.infer_config(
            model,
            lambda msg: {"num_samples": args.tmc_num_samples, "expand": False} if msg["infer"].get("enumerate", None) == "parallel" else {})  # noqa: E501
        svi = SVI(tmc_model, guide, optim, elbo)
    else:
        elbo = TraceEnum_ELBO(max_plate_nesting=1, num_particles=20, vectorize_particles=True)
        svi = SVI(model, guide, optim, elbo)

    losses = []

    print("Beginning training of model_{} with Stochastic Variational Inference.".format(args.model))

    for step in range(args.num_steps):
        loss = svi.step(capture_history, sex)
        losses.append(loss)
        if step % 20 == 0 and step > 0 or step == args.num_steps - 1:
            print("[iteration %03d] loss: %.3f" % (step, np.mean(losses[-20:])))

    # evaluate final trained model
    elbo_eval = TraceEnum_ELBO(max_plate_nesting=1, num_particles=2000, vectorize_particles=True)
    svi_eval = SVI(model, guide, optim, elbo_eval)
    print("Final loss: %.4f" % svi_eval.evaluate_loss(capture_history, sex))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CJS capture-recapture model for ecological data")
    parser.add_argument("-m", "--model", default="1", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-d", "--dataset", default="dipper", type=str)
    parser.add_argument("-n", "--num-steps", default=400, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.002, type=float)
    parser.add_argument("--tmc", action='store_true',
                        help="Use Tensor Monte Carlo instead of exact enumeration "
                             "to estimate the marginal likelihood. You probably don't want to do this, "
                             "except to see that TMC makes Monte Carlo gradient estimation feasible "
                             "even with very large numbers of non-reparametrized variables.")
    parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    main(args)
