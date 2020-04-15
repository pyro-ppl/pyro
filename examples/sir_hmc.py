# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, config_enumerate
from pyro.infer.reparam import DiscreteCosineReparam

logging.basicConfig(format='%(message)s', level=logging.INFO)


# First consider a simple SIR model (Susceptible Infected Recovered).
#
# Note we need to use ExtendedBinomial rather than Binomial because the data
# may lie outside of the predicted support. For these values,
# Binomial.log_prob() will error, whereas ExtendedBinomial.log_prob() will
# return -inf.

def discrete_model(data, population=10000):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))
    population = 10000

    # Sequentially filter.
    S = torch.tensor(population - 1.)
    I = torch.tensor(1.)
    for t, datum in enumerate(data):
        S2I = pyro.sample("S2I_{}".format(t),
                          dist.Binomial(S, prob_s * I / population))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(I, prob_i))
        S = S - S2I
        I = I + S2I - I2R
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(I, rho),  # See Note 1 above.
                    obs=datum)


# We can use this model to simulate data. We'll use poutine.condition to pin
# parameter values and poutine.trace to record sample observations.

def generate_data(args):
    params = {"probs_s": torch.tensor(args.infection_rate),
              "probs_i": torch.tensor(args.recovery_rate),
              "rho": torch.tensor(args.response_rate)}
    empty_data = [None] * args.duration

    with poutine.trace() as tr:
        with poutine.condition(data=params):
            discrete_model(empty_data, args.population)

    data = torch.stack([site["value"]
                        for site in tr.trace.nodes.values()
                        if site["name"].startswith("obs_")])
    logging.info("Generated data:\n{}".format(" ".join([str(int(x)) for x in data])))
    return data


# Consider reparameterizing in terms of the variables (S, I) rather than (S2I,
# I2R). Since these may lead to inconsistent states, we need to replace the
# Binomial transition factors (S2I, I2R) with ExtendedBinomial.
#
# The following model is equivalent:

@config_enumerate
def reparameterized_discrete_model(data, population=10000):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))

    # Sequentially filter.
    S = torch.tensor(population - 1.)
    I = torch.tensor(1.)
    for t, datum in enumerate(data):
        # Sample reparametrizing variables.
        # Note the density is ignored; distributions are used only for initialization.
        with poutine.mask(mask=False):
            S_next = pyro.sample("S_{}".format(t), dist.Binomial(population, 0.5))
            I_next = pyro.sample("I_{}".format(t), dist.Binomial(population, 0.5))

        # Now we reverse the computation.
        S2I = S - S_next
        I2R = I - I_next + S2I
        pyro.sample("S2I_{}".format(t),
                    dist.ExtendedBinomial(S, prob_s * I / population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(I, prob_i),
                    obs=I2R)
        S = S_next
        I = I_next
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(I, rho),
                    obs=datum)


# By reparameterizing, we have converted to coordinates to make the model
# Markov. We have also replaced dynamic integer_interval constraints with
# easier static integer_interval constraints (although we'll still need good
# initialization to avoid NANs). Since the discrete latent variables are
# bounded (by population size), we can enumerate out discrete latent variables
# and perform HMC inference over the global latents. However enumeration
# complexity is O(population^4), so this is only feasible for very small
# populations.
#
# Here is an inference approch using the NUTS sampler.

def infer_enum(args, data):
    model = reparameterized_discrete_model
    kernel = NUTS(model, jit_compile=args.jit, ignore_jit_warnings=True)
    mcmc = MCMC(kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps)
    mcmc.run(data)


# To perform exact inference on large populations, we'll continue to
# reparameterize, this time replacing each of (S_aux,I_aux) with a combination of
# a positive real variable and a Bernoulli variable.
#
# This is the crux: we can now perform HMC over the real variable and marginalize
# out the Bernoulli using variable elimination.
#
# We first define a helper to create enumerated Bernoulli sites.

def quantize(name, x_real):
    """Randomly quantize in a way that preserves probability mass."""
    # Note this is a linear spline, but we could easily replace with a
    # quadratic spline to ensure gradients are continuous.
    lb = x_real.floor()
    bern = pyro.sample(name, dist.Bernoulli(x_real - lb))
    return lb + bern


# Now we can define another equivalent model.

@config_enumerate
def continuous_model(data, population=10000):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))

    # Sample reparametrizing variables.
    with pyro.plate("time", len(data)):
        S_aux = pyro.sample("S_aux", dist.Uniform(-0.5, population + 0.5))
        I_aux = pyro.sample("I_aux", dist.Uniform(-0.5, population + 0.5))

    # Sequentially filter.
    S = torch.tensor(population - 1.)
    I = torch.tensor(1.)
    for t, datum in poutine.markov(enumerate(data)):
        S_next = quantize("S_{}".format(t), S_aux[t]).clamp(min=0, max=population)
        I_next = quantize("I_{}".format(t), I_aux[t]).clamp(min=0, max=population)

        # Now we reverse the computation.
        S2I = S - S_next
        I2R = I - I_next + S2I
        # See Note 1 above.
        pyro.sample("S2I_{}".format(t), dist.Binomial(S, prob_s * I / population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t), dist.Binomial(I, prob_i),
                    obs=I2R)
        S = S_next
        I = I_next
        # See Note 1 above.
        pyro.sample("obs_{}".format(t), dist.Binomial(I, rho),
                    obs=datum)


# Now all latent variables in the continuous_model are either continuous or
# enumerated, so we can use HMC.
#
# One trick to improve inference geometry is to reparameterize the S_aux,I_aux
# variables via DiscreteCosineReparam. This especially allows HMC's diagonal mass
# matrix to learn different step sizes for high- and low-frequency directions.
# We can apply that outside of the model, during inference.

def infer_cont(args, data):
    model = continuous_model
    if args.dct:
        model = poutine.reparam(model, {"S_aux": DiscreteCosineReparam(),
                                        "I_aux": DiscreteCosineReparam()})

    kernel = NUTS(model, jit_compile=args.jit, ignore_jit_warnings=True)
    mcmc = MCMC(kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps)
    mcmc.run(data)


# The next step is to vectorize. We can repurpose DiscreteHMM here, but we'll
# need to manually represent a Markov neighborhood of multiple Bernoullis as
# single joint Categorical with 2 x 2 = 4 states. We leave this to future work.

# Finally we'll define an experiment runner.

def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed)

    data = generate_data(args)
    if args.enum:
        infer_enum(args, data)
    else:
        infer_cont(args, data)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    parser = argparse.ArgumentParser(description="SIR outbreak modeling using HMC")
    parser.add_argument("--population", default=10, type=int)
    parser.add_argument("--duration", default=10, type=int)
    parser.add_argument("--infection-rate", default=0.2, type=float)
    parser.add_argument("--recovery-rate", default=0.2, type=float)
    parser.add_argument("--response-rate", default=0.2, type=float)
    parser.add_argument("--enum", action="store_true", default=False,
                        help="use the full enumeration model")
    parser.add_argument("--dct", action="store_true", default=False,
                        help="use discrete cosine reparameterizer")
    parser.add_argument("-n", "--num-samples", default=200, type=int)
    parser.add_argument("--warmup-steps", default=100, type=int)
    parser.add_argument("--rng-seed", nargs='?', default=0, type=int)
    parser.add_argument("--jit", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()

    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    main(args)
