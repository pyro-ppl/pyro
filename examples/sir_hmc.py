# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import torch
from torch.distributions import biject_to, constraints
from torch.distributions.transforms import ComposeTransform

import pyro
import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.distributions.transforms.discrete_cosine import DiscreteCosineTransform
from pyro.infer import MCMC, NUTS, config_enumerate
from pyro.infer.autoguide import init_to_value
from pyro.infer.reparam import DiscreteCosineReparam
from pyro.ops.tensor_utils import convolve, safe_log
from pyro.util import warn_if_nan

logging.basicConfig(format='%(message)s', level=logging.INFO)


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

    # We'll retry until we get an actual outbreak.
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
        # Sample reparameterizing variables.
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
    return _infer_hmc(args, data, model)


def _infer_hmc(args, data, model, init_values={}):
    kernel = NUTS(model,
                  max_tree_depth=args.max_tree_depth,
                  init_strategy=init_to_value(values=init_values),
                  jit_compile=args.jit, ignore_jit_warnings=True)

    def logging_hook(kernel, *args):
        print("potential = {:0.6g}".format(kernel._potential_energy_last))

    mcmc = MCMC(kernel,
                hook_fn=logging_hook if args.verbose else None,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps)
    mcmc.run(data, population=args.population)
    samples = mcmc.get_samples()
    return samples


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

    # Sample reparameterizing variables.
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


# Now all latent variables in the continuous_model are either continuous or
# enumerated, so we can use HMC.
#
# One trick to improve inference geometry is to reparameterize the S_aux,I_aux
# variables via DiscreteCosineReparam. This especially allows HMC's diagonal mass
# matrix to learn different step sizes for high- and low-frequency directions.
# We can apply that outside of the model, during inference.

def infer_hmc_cont(model, args, data):
    if args.dct:
        model = poutine.reparam(model, {"S_aux": DiscreteCosineReparam(),
                                        "I_aux": DiscreteCosineReparam()})

    # Heuristically initialize to a good feasible point.
    S0 = args.population - 1  # Start with a single infection.
    S2I = data * min(2., S0 / data.sum())  # Assume 50% response rate.
    S2I[0] += 1  # Account for the single infection.
    S_aux = (S0 - S2I.cumsum(-1)).clamp_(min=0.5)
    # Assume infection lasts a few days.
    I_aux = convolve(S2I, torch.tensor([1., 0.8, 0.5, 0.3, 0.2]))[:len(data)].clamp_(min=0.5)
    # Also initialze DCT coordinates.
    t = ComposeTransform([biject_to(constraints.interval(-0.5, 0.5 + args.population)).inv,
                          DiscreteCosineTransform(dim=-1)])
    init_values = {
        "prob_s": torch.tensor(0.5),
        "prob_i": torch.tensor(0.5),
        "rho": torch.tensor(0.5),
        "S_aux": S_aux,
        "I_aux": I_aux,
        "S_aux_dct": t(S_aux),
        "I_aux_dct": t(I_aux),
    }

    return _infer_hmc(args, data, model, init_values=init_values)


# The next step is to vectorize. We can repurpose DiscreteHMM's implementation
# here, but we'll need to manually represent a Markov neighborhood of multiple
# Bernoullis as single joint Categorical with 2 x 2 = 4 states.

def quantize_enumerate(x_real, min, max):
    """Quantize, then manually enumerate."""
    lb = x_real.detach().floor()
    # This cubic spline guarantees gradients are continuous.
    t = x_real - lb
    prob = t * t * (3 - 2 * t)
    x = torch.stack([lb, lb + 1], dim=-1).clamp(min=min, max=max)
    logits = safe_log(torch.stack([1 - prob, prob], dim=-1))
    return x, logits


def vectorized_model(data, population):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))

    # Sample reparameterizing variables.
    S_aux = pyro.sample("S_aux",
                        dist.Uniform(-0.5, population + 0.5)
                            .expand(data.shape).to_event(1))
    I_aux = pyro.sample("I_aux",
                        dist.Uniform(-0.5, population + 0.5)
                            .expand(data.shape).to_event(1))

    # Manually enumerate.
    S_curr, S_logp = quantize_enumerate(S_aux, min=0, max=population)
    I_curr, I_logp = quantize_enumerate(I_aux, min=0, max=population)
    S_prev = torch.nn.functional.pad(S_curr[:-1], (0, 0, 1, 0), value=population - 1)
    I_prev = torch.nn.functional.pad(I_curr[:-1], (0, 0, 1, 0), value=1)
    # Reshape to support broadcasting, similar EnumMessenger.
    S_prev = S_prev.reshape(-1, 2, 1, 1, 1)
    I_prev = I_prev.reshape(-1, 1, 2, 1, 1)
    S_curr = S_curr.reshape(-1, 1, 1, 2, 1)
    S_logp = S_logp.reshape(-1, 1, 1, 2, 1)
    I_curr = I_curr.reshape(-1, 1, 1, 1, 2)
    I_logp = S_logp.reshape(-1, 1, 1, 1, 2)
    data = data.reshape(-1, 1, 1, 1, 1)

    # Reverse the S2I,I2R computation.
    S2I = S_prev - S_curr
    I2R = I_prev - I_curr + S2I

    # Compute probability factors.
    S2I_logp = dist.ExtendedBinomial(S_prev, prob_s * I_prev / population).log_prob(S2I)
    I2R_logp = dist.ExtendedBinomial(I_prev, prob_i).log_prob(I2R)
    obs_logp = dist.ExtendedBinomial(S2I.clamp(min=0), rho).log_prob(data)

    # Manually perform variable elimination.
    logp = S_logp + (I_logp + obs_logp) + S2I_logp + I2R_logp
    logp = logp.reshape(-1, 2 * 2, 2 * 2)
    logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)
    logp = logp.logsumexp([0, 1])
    warn_if_nan(logp)
    pyro.factor("obs", logp)


# We can train fit vectorized_model exactly as we fit the original
# continuous_model.
#
# Finally we'll define an experiment runner.

def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed)

    data = generate_data(args)

    # Choose among inference methods.
    if args.enum:
        samples = infer_hmc_enum(args, data)
    elif args.vectorized:
        samples = infer_hmc_cont(vectorized_model, args, data)
    else:
        samples = infer_hmc_cont(continuous_model, args, data)

    # Evaluate results.
    names = {"response_rate": "rho",
             "infection_rate": "prob_s",
             "recovery_rate": "prob_i"}
    for name, key in names.items():
        mean = samples[key].mean().item()
        std = samples[key].std().item()
        logging.info("{}: truth = {:0.3g}, estimate = {:0.3g} \u00B1 {:0.3g}"
                     .format(name, getattr(args, name), mean, std))

    return samples


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    parser = argparse.ArgumentParser(description="SIR outbreak modeling using HMC")
    parser.add_argument("-p", "--population", default=10, type=int)
    parser.add_argument("-d", "--duration", default=10, type=int)
    parser.add_argument("--infection-rate", default=0.3, type=float)
    parser.add_argument("--recovery-rate", default=0.3, type=float)
    parser.add_argument("--response-rate", default=0.5, type=float)
    parser.add_argument("--enum", action="store_true",
                        help="use the full enumeration model")
    parser.add_argument("-v", "--vectorized", action="store_true",
                        help="use the vectorized continuous model")
    parser.add_argument("--dct", action="store_true",
                        help="use discrete cosine reparameterizer")
    parser.add_argument("-n", "--num-samples", default=200, type=int)
    parser.add_argument("-w", "--warmup-steps", default=100, type=int)
    parser.add_argument("--max-tree-depth", default=4, type=int)
    parser.add_argument("--rng-seed", nargs='?', default=0, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    main(args)
