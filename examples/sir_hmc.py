# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math

import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import HMC, MCMC, NUTS, SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.infer.autoguide import AutoNormal
from pyro.infer.reparam import DiscreteCosineReparam
from pyro.optim import ClippedAdam

logging.basicConfig(format='%(message)s', level=logging.INFO)


def print_dot(*args, **kwargs):
    import sys
    sys.stderr.write(".")
    sys.stderr.flush()


# First consider a simple SIR model (Susceptible Infected Recovered).
#
# Note we need to use ExtendedBinomial rather than Binomial because the data
# may lie outside of the predicted support. For these values,
# Binomial.log_prob() will error, whereas ExtendedBinomial.log_prob() will
# return -inf.

def discrete_model(data, population):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))

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
                    dist.ExtendedBinomial(S2I, rho),  # See Note 1 above.
                    obs=datum)


# We can use this model to simulate data. We'll use poutine.condition to pin
# parameter values and poutine.trace to record sample observations.

def generate_data(args):
    params = {"probs_s": torch.tensor(args.infection_rate),
              "probs_i": torch.tensor(args.recovery_rate),
              "rho": torch.tensor(args.response_rate)}
    empty_data = [None] * args.duration

    # We'lll retry until we get an actual outbreak.
    for attempt in range(100):
        with poutine.trace() as tr:
            with poutine.condition(data=params):
                discrete_model(empty_data, args.population)

        data = torch.stack([site["value"]
                            for site in tr.trace.nodes.values()
                            if site["name"].startswith("obs_")])
        if data.sum() > 2:
            break
    logging.info("Generated data:\n{}".format(" ".join([str(int(x)) for x in data])))
    return data


# Consider reparameterizing in terms of the variables (S, I) rather than (S2I,
# I2R). Since these may lead to inconsistent states, we need to replace the
# Binomial transition factors (S2I, I2R) with ExtendedBinomial.
#
# The following model is equivalent:

@config_enumerate
def reparameterized_discrete_model(data, population):
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
                    dist.ExtendedBinomial(S2I.clamp(min=0), rho),
                    obs=datum)
    print_dot()


# By reparameterizing, we have converted to coordinates to make the model
# Markov. We have also replaced dynamic integer_interval constraints with
# easier static integer_interval constraints (although we'll still need good
# initialization to avoid NANs). Since the discrete latent variables are
# bounded (by population size), we can enumerate out discrete latent variables
# and perform HMC inference over the global latents. However enumeration
# complexity is O(population^4), so this is only feasible for very small
# populations.
#
# Here is an inference approch using an MCMC sampler.

def infer_hmc_enum(args, data):
    model = reparameterized_discrete_model
    _infer_hmc(args, data, model)


def _infer_hmc(args, data, model, init_params=None):
    Kernel = NUTS if args.nuts else HMC
    kernel = Kernel(model, jit_compile=args.jit, ignore_jit_warnings=True)
    mcmc = MCMC(kernel,
                initial_params=init_params,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                hook_fn=print_dot)
    mcmc.run(data, population=args.population)
    samples = mcmc.get_samples()
    for name, value in samples.items():
        if value.shape[1:].numel() == 1:
            logging.info("median {} = {:0.3g}".format(name, value.median(0).values.item()))


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
    lb = x_real.detach().floor()
    # This cubic spline guarantees gradients are continuous.
    t = x_real - lb
    prob = t * t * (3 - 2 * t)
    bern = pyro.sample(name, dist.Bernoulli(prob))
    return lb + bern


# Now we can define another equivalent model.

@config_enumerate
def continuous_model(data, population):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))

    # Sample reparametrizing variables.
    S_aux = pyro.sample("S_aux",
                        dist.Uniform(-0.5, population + 0.5)
                            .expand(data.shape).to_event(1))
    I_aux = pyro.sample("I_aux",
                        dist.Uniform(-0.5, population + 0.5)
                            .expand(data.shape).to_event(1))

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
        pyro.sample("S2I_{}".format(t),
                    dist.ExtendedBinomial(S, prob_s * I / population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(I, prob_i),
                    obs=I2R)
        S = S_next
        I = I_next
        # See Note 1 above.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I.clamp(min=0), rho),
                    obs=datum)
    print_dot()


# Now all latent variables in the continuous_model are either continuous or
# enumerated, so we can use HMC.
#
# One trick to improve inference geometry is to reparameterize the S_aux,I_aux
# variables via DiscreteCosineReparam. This especially allows HMC's diagonal mass
# matrix to learn different step sizes for high- and low-frequency directions.
# We can apply that outside of the model, during inference.

def infer_hmc_cont(args, data):
    model = continuous_model
    if args.dct:
        model = poutine.reparam(model, {"S_aux": DiscreteCosineReparam(),
                                        "I_aux": DiscreteCosineReparam()})

    # Note these are in unconstrained space.
    if args.dct:
        init_params = None  # TODO
    else:
        t = biject_to(constraints.interval(-0.5, args.population + 0.5)).inv
        init_params = {
            "prob_s": torch.tensor(0.),
            "prob_i": torch.tensor(0.),
            "rho": torch.tensor(0.),
            "S_aux": t(args.population - 0.5 - torch.arange(float(args.duration))),
            "I_aux": t(torch.full((args.duration,), 1.5)),
        }

    _infer_hmc(args, data, model, init_params=init_params)


# Alternatively we could perform variational inference, which is approximate.

def infer_svi_cont(args, data):
    model = continuous_model
    if args.dct:
        model = poutine.reparam(model, {"S_aux": DiscreteCosineReparam(),
                                        "I_aux": DiscreteCosineReparam()})

    continuous_sites = ["probs_s", "probs_i", "rho", "S_aux", "I_aux"]
    guide = AutoNormal(poutine.block(model, expose=continuous_sites),
                       init_scale=0.01)
    optim = ClippedAdam({"lr": 0.01})
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    svi = SVI(model, guide, optim, Elbo())
    for step in range(1001):
        loss = svi.step(data, population=args.population)
        assert not math.isnan(loss)
        if step % 50 == 0:
            logging.info("step {: >4g} loss = {:0.4g}".format(step, loss))

    for name, value in guide.median():
        if value.numel() == 1:
            logging.info("median {} = {:0.3g}".format(name, value.item()))


# The next step is to vectorize. We can repurpose DiscreteHMM here, but we'll
# need to manually represent a Markov neighborhood of multiple Bernoullis as
# single joint Categorical with 2 x 2 = 4 states. We leave this to future work.

# Finally we'll define an experiment runner.

def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed)

    data = generate_data(args)

    # Choose among inference methods.
    if args.enum:
        if args.svi:
            raise NotImplementedError
        else:
            infer_hmc_enum(args, data)
    else:
        if args.svi:
            infer_svi_cont(args, data)
        else:
            infer_hmc_cont(args, data)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    parser = argparse.ArgumentParser(description="SIR outbreak modeling using HMC")
    parser.add_argument("--population", default=10, type=int)
    parser.add_argument("--duration", default=10, type=int)
    parser.add_argument("--infection-rate", default=0.3, type=float)
    parser.add_argument("--recovery-rate", default=0.3, type=float)
    parser.add_argument("--response-rate", default=0.5, type=float)
    parser.add_argument("--enum", action="store_true", default=False,
                        help="use the full enumeration model")
    parser.add_argument("--nuts", action="store_true", default=False,
                        help="use NUTS rather than HMC for inference")
    parser.add_argument("--svi", action="store_true", default=False,
                        help="use SVI for inference in the complete model")
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
