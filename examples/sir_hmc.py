# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import re

import torch
from torch.autograd import grad
from torch.distributions import biject_to, constraints
from torch.distributions.transforms import ComposeTransform

import pyro
import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.distributions.transforms.discrete_cosine import DiscreteCosineTransform
from pyro.infer import MCMC, NUTS, config_enumerate, infer_discrete
from pyro.infer.autoguide import init_to_value
from pyro.infer.mcmc.util import TraceEinsumEvaluator
from pyro.infer.reparam import DiscreteCosineReparam
from pyro.ops.tensor_utils import convolve, safe_log
from pyro.util import warn_if_nan

logging.basicConfig(format='%(message)s', level=logging.INFO)


# Introduction
# ============
#
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
        S = pyro.deterministic("S_{}".format(t), S - S2I)
        I = pyro.deterministic("I_{}".format(t), I + S2I - I2R)
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
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


# Inference
# =========
#
# While the above discrete_model is easy to understand, its discrete latent
# variables pose a challenge for inference. One of the most popular inference
# strategies for such models is Sequential Monte Carlo. However since Pyro and
# PyTorch are stronger in gradient based vectorizable inference algorithms, we
# will instead pursue inference based on Hamiltonian Monte Carlo (HMC).
#
# Our general inference strategy will be to:
# 1. Introduce auxiliary variables to make the model Markov.
# 2. Introduce more auxiliary variables to create a discrete parameterization.
# 3. Marginalize out all remaining discrete latent variables.
# 4. Vectorize to enable parallel-scan temporal filtering.
#
# Let's consider reparameterizing in terms of the variables (S, I) rather than
# (S2I, I2R). Since these may lead to inconsistent states, we need to replace
# the Binomial transition factors (S2I, I2R) with ExtendedBinomial.
#
# The following model is equivalent to the discrete_model:

@config_enumerate
def reparameterized_discrete_model(data, population):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))

    # Sequentially filter.
    S_curr = torch.tensor(population - 1.)
    I_curr = torch.tensor(1.)
    for t, datum in enumerate(data):
        # Sample reparameterizing variables.
        # Note the density is ignored; distributions are used only for
        # initialization.
        with poutine.mask(mask=False):
            S_prev, I_prev = S_curr, I_curr
            S_curr = pyro.sample("S_{}".format(t), dist.Binomial(population, 0.5))
            I_curr = pyro.sample("I_{}".format(t), dist.Binomial(population, 0.5))

        # Now we reverse the computation.
        S2I = S_prev - S_curr
        I2R = I_prev - I_curr + S2I
        pyro.sample("S2I_{}".format(t),
                    dist.ExtendedBinomial(S_prev, prob_s * I_prev / population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(I_prev, prob_i),
                    obs=I2R)
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
# Here is an inference approach using an MCMC sampler.

def infer_hmc_enum(args, data):
    model = reparameterized_discrete_model
    return _infer_hmc(args, data, model)


def _infer_hmc(args, data, model, init_values={}):
    kernel = NUTS(model,
                  max_tree_depth=args.max_tree_depth,
                  init_strategy=init_to_value(values=init_values),
                  jit_compile=args.jit, ignore_jit_warnings=True)
    energies = []

    def hook_fn(kernel, *unused):
        e = float(kernel._potential_energy_last)
        energies.append(e)
        if args.verbose:
            print("potential = {:0.6g}".format(e))

    mcmc = MCMC(kernel, hook_fn=hook_fn,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps)
    mcmc.run(data, population=args.population)
    mcmc.summary()
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(energies)
        plt.xlabel("MCMC step")
        plt.ylabel("potential energy")
        plt.tight_layout()

    samples = mcmc.get_samples()
    return samples


# To scale to large populations, we'll continue to reparameterize, this time
# replacing each of (S_aux,I_aux) with a combination of a positive real
# variable and a Bernoulli variable.
#
# This is the crux: we can now perform HMC over the real variable and
# marginalize out the Bernoulli using variable elimination.
#
# We first define a helper to create enumerated Bernoulli sites.

SPLINE_ORDER = 3


def quantize(name, x_real, min, max):
    """Randomly quantize in a way that preserves probability mass."""
    lb = x_real.detach().floor()
    # This cubic spline guarantees gradients are continuous.
    t = x_real - lb
    prob = t * t * (3 - 2 * t) if SPLINE_ORDER == 3 else t
    bern = pyro.sample("Q_" + name, dist.Bernoulli(prob))
    x = (lb + bern).clamp(min=min, max=max)
    return pyro.deterministic(name, x)


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
    S_curr = torch.tensor(population - 1.)
    I_curr = torch.tensor(1.)
    for t, datum in poutine.markov(enumerate(data)):
        S_prev, I_prev = S_curr, I_curr
        S_curr = quantize("S_{}".format(t), S_aux[..., t], min=0, max=population)
        I_curr = quantize("I_{}".format(t), I_aux[..., t], min=0, max=population)

        # Now we reverse the computation.
        S2I = S_prev - S_curr
        I2R = I_prev - I_curr + S2I
        pyro.sample("S2I_{}".format(t),
                    dist.ExtendedBinomial(S_prev, prob_s * I_prev / population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(I_prev, prob_i),
                    obs=I2R)
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I.clamp(min=0), rho),
                    obs=datum)


# Now all latent variables in the continuous_model are either continuous or
# enumerated, so we can use HMC. However we need to take special care with
# constraints because the above Markov reparameterization covers regions of
# hypothesis space that are infeasible (i.e. whose log_prob is -infinity). We
# thus heuristically initialize to a feasible point.

def heuristic_init(args, data):
    """Heuristically initialize to a feasible point."""
    # Start with a single infection.
    S0 = args.population - 1
    if args.heuristic == 1:
        # Assume 50% <= response rate <= 100%.
        S2I = data * min(2., (S0 / data.sum()).sqrt())
        S_aux = (S0 - S2I.cumsum(-1)).clamp(min=0.5)
        # Account for the single initial infection.
        S2I[0] += 1
        # Assume infection lasts less than a month.
        recovery = (1 - args.recovery_rate / 2) ** torch.arange(30.)
        I_aux = convolve(S2I, recovery)[:len(data)].clamp(min=0.5)
    elif args.heuristic == 2:
        # Assume 100% response rate, 0% recovery rate.
        S_aux = S0 - data.cumsum(-1)
        I_aux = 1 + data.cumsum(-1)
    # Also initialize DCT transformed coordinates.
    t = ComposeTransform([biject_to(constraints.interval(-0.5, args.population + 0.5)).inv,
                          DiscreteCosineTransform(dim=-1)])

    return {
        "prob_s": torch.tensor(0.5),
        "prob_i": torch.tensor(0.5),
        "rho": torch.tensor(0.5),
        "S_aux": S_aux,
        "I_aux": I_aux,
        "S_aux_dct": t(S_aux),
        "I_aux_dct": t(I_aux),
    }


# One trick to improve inference geometry is to reparameterize the S_aux,I_aux
# variables via DiscreteCosineReparam. This allows HMC's diagonal mass matrix
# adaptation to learn different step sizes for high- and low-frequency
# directions. We can apply that outside of the model, during inference.

def infer_hmc_cont(model, args, data):
    if args.dct:
        model = poutine.reparam(model, {"S_aux": DiscreteCosineReparam(),
                                        "I_aux": DiscreteCosineReparam()})
    init_values = heuristic_init(args, data)
    return _infer_hmc(args, data, model, init_values=init_values)


# Our final inference trick is to vectorize. We can repurpose DiscreteHMM's
# implementation here, but we'll need to manually represent a Markov
# neighborhood of multiple Bernoullis as single joint Categorical with 2 * 2 =
# 4 states, and them manually perform variable elimination (the factors here
# don't quite conform to DiscreteHMM's interface).

def quantize_enumerate(x_real, min, max):
    """Quantize, then manually enumerate."""
    lb = x_real.detach().floor()
    # This cubic spline guarantees gradients are continuous.
    t = x_real - lb
    prob = t * t * (3 - 2 * t) if SPLINE_ORDER == 3 else t
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
    I_logp = I_logp.reshape(-1, 1, 1, 1, 2)
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
    logp = logp.reshape(-1).logsumexp(0)
    logp = logp - math.log(4)  # Account for S,I initial distributions.
    warn_if_nan(logp)
    pyro.factor("obs", logp)


# We can fit vectorized_model exactly as we fit the original continuous_model,
# using our infer_hmc_cont helper. The vectorized model is more than an order
# of magnitude faster than the sequential version, and scales logarithmically
# in time (up to your machine's parallelism).


# Prediction and Forecasting
# ==========================
#
# So far we've written four models that each describe the same probability
# distribution. Each successive model made inference cheaper. Next let's move
# beyond inference and consider predicting latent infection rate and
# forecasting future infections.
#
# We'll use Pyro's effect handlers to combine multiple of the above models,
# leveraging the vectorized_model for inference, then the continuous_model to
# compute local latent variables, and finally the original discrete_model to
# forecast forward in time. Let's assume posterior samples have already been
# generated via infer_hmc_cont(vectorized_model, ...).

@torch.no_grad()
def predict(args, data, samples):
    particle_plate = pyro.plate("particles", args.num_samples, dim=-1)

    # First we sample discrete auxiliary variables from the continuous
    # variables sampled in vectorized_model. Here infer_discrete runs a
    # forward-filter backward-sample algorithm. We'll add these new samples to
    # the existing dict of samples.
    model = infer_discrete(particle_plate(continuous_model),
                           first_available_dim=-2)
    with poutine.trace() as tr, \
            poutine.condition(data=samples):
        model(data, args.population)
    samples = {name: site["value"]
               for name, site in tr.trace.nodes.items()
               if site["type"] == "sample"}

    # Next we'll run the forward generative process in discrete_model. Again
    # we'll update the dict of samples.
    extended_data = list(data)
    extended_data.extend([None] * args.forecast)
    with poutine.trace() as tr, \
            poutine.condition(data=samples), \
            particle_plate:
        discrete_model(extended_data, args.population)
    samples = {name: site["value"]
               for name, site in tr.trace.nodes.items()
               if site["type"] == "sample"}

    # Concatenate sequential time series into tensors.
    for key in ("S", "I", "S2I", "I2R"):
        pattern = key + "_[0-9]+"
        series = [value
                  for name, value in samples.items()
                  if re.match(pattern, name)]
        assert len(series) == args.duration + args.forecast
        samples[key] = torch.stack(series, dim=-1)

    # Optionally plot the latent and forecasted series of new infections.
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        time = torch.arange(len(data))
        S2I = samples["S2I"]
        p50 = S2I.median(0).values
        p05 = S2I.kthvalue(int(round(0.05 * args.num_samples))).values
        p95 = S2I.kthvalue(int(round(0.95 * args.num_samples))).values
        plt.plot(time, data, "k-", label="observed")
        plt.plot(time, p50, "r-", label="median")
        plt.fill_between(time, p05, p95, color="red", alpha=0.3, label="90% CI")
        plt.axvline(args.duration, color="gray")
        plt.xlabel("day after first infection")
        plt.ylabel("new infections per day")
        plt.title("Predicted new infections")
        plt.legend(loc="best")
        plt.tight_layout()

    return samples


# Experiments
# ===========
#
# Finally we'll define an experiment runner.

def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed)

    data = generate_data(args)

    # Optionally test initialization heuristic.
    if args.test_init:
        _test_init(args, data)
        return

    # Choose among inference methods.
    if args.enum:
        samples = infer_hmc_enum(args, data)
    elif args.vectorized:
        samples = infer_hmc_cont(vectorized_model, args, data)
    else:
        samples = infer_hmc_cont(continuous_model, args, data)

    # Evaluate fit.
    names = {"response_rate": "rho",
             "infection_rate": "prob_s",
             "recovery_rate": "prob_i"}
    for name, key in names.items():
        mean = samples[key].mean().item()
        std = samples[key].std().item()
        logging.info("{}: truth = {:0.3g}, estimate = {:0.3g} \u00B1 {:0.3g}"
                     .format(name, getattr(args, name), mean, std))

    # Predict latent time series.
    if args.forecast:
        samples = predict(args, data, samples)

    return samples


def _test_init(args, data):
    """Test helper to debug the init heuristic."""
    init_values = heuristic_init(args, data)
    logging.info("-" * 40)
    for name, x in init_values.items():
        logging.info("{}:{}{}".format(name, "\n" if x.shape else " ", x))
        x.requires_grad_()
    model = vectorized_model if args.vectorized else continuous_model
    with poutine.trace() as tr, poutine.condition(data=init_values):
        model(data, args.population)

    # Test log prob.
    logging.info("-" * 40)
    if args.vectorized:
        log_prob = tr.trace.log_prob_sum()
    else:
        log_prob = TraceEinsumEvaluator(tr.trace, True, 0).log_prob(tr.trace)
    logging.info("log_prob = {:0.6g}".format(log_prob))
    if not torch.isfinite(log_prob):
        raise ValueError("infinite log_prob")

    # Test gradients.
    grads = grad(log_prob, list(init_values.values()), allow_unused=True)
    for name, dx in zip(init_values, grads):
        if dx is None:
            logging.info("{}: no gradient".format(name))
            continue
        logging.info("d/d {}:{}{}".format(name, "\n" if dx.shape else " ", dx))
        if not torch.isfinite(dx).all():
            raise ValueError("invalid gradient for {}".format(name))

    # Smoke test.
    logging.info("-" * 40)
    args.warmup_steps = 0
    args.num_samples = 1
    args.max_tree_depth = 1
    infer_hmc_cont(model, args, data)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    parser = argparse.ArgumentParser(description="SIR outbreak modeling using HMC")
    parser.add_argument("-p", "--population", default=10, type=int)
    parser.add_argument("-d", "--duration", default=10, type=int)
    parser.add_argument("-f", "--forecast", default=0, type=int)
    parser.add_argument("--infection-rate", default=0.3, type=float)
    parser.add_argument("--recovery-rate", default=0.3, type=float)
    parser.add_argument("--response-rate", default=0.5, type=float)
    parser.add_argument("-i", "--heuristic", default=1, type=int)
    parser.add_argument("--enum", action="store_true",
                        help="use the full enumeration model")
    parser.add_argument("--spline-order", default=3, type=int)
    parser.add_argument("--dct", action="store_true",
                        help="use discrete cosine reparameterizer")
    parser.add_argument("-v", "--vectorized", action="store_true",
                        help="use the vectorized continuous model")
    parser.add_argument("-n", "--num-samples", default=200, type=int)
    parser.add_argument("-w", "--warmup-steps", default=100, type=int)
    parser.add_argument("-t", "--max-tree-depth", default=4, type=int)
    parser.add_argument("-s", "--rng-seed", default=0, type=int)
    parser.add_argument("--test-init", action="store_true")
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.double:
        if args.cuda:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
    elif args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    SPLINE_ORDER = args.spline_order

    main(args)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
