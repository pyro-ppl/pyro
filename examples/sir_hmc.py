# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate


"""
First consider a simple SIR model.

Note 1. Even in this simple model we would need to fix Binomial.log_prob() to
    return -inf for values outside of its support.
"""


def discrete_model(data):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))
    population = 10000

    # Sequentially filter.
    S = torch.tensor(population - 1.)
    I = torch.tensor(1.)
    for t, datum in enumerate(data):
        S2I = pyro.sample("S2I_{}".format(t), dist.Binomial(S, prob_s * I / population))
        I2R = pyro.sample("I2R_{}".format(t), dist.Binomial(I, prob_i))
        S = S - S2I
        I = I + S2I - I2R
        # See Note 1 above.
        pyro.sample("obs_{}".format(t), dist.Binomial(I, rho),
                    obs=datum)


"""
Consider reparameterizing in terms of the variables (S_aux, I_aux) rather than
(S2I, I2R). The following model is equivalent:
"""


def reparameterized_discrete_model(data):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))
    population = 10000

    # Sample reparametrizing variables.
    with pyro.plate("time", len(data)):
        # Note the density is ignored; distributions are used only for initialization.
        with poutine.mask(mask=False):
            S_aux = pyro.sample("S_aux", dist.Geometric(0.1))
            I_aux = pyro.sample("I_aux", dist.Geometric(0.1))

    # Sequentially filter.
    S = torch.tensor(population - 1.)
    I = torch.tensor(1.)
    for t, datum in enumerate(data):
        S_next = S_aux[t]
        I_next = I_aux[t]

        # Now we reverse the computation.
        S2I = S - S_next
        I2R = I - I_next + S2I
        # See Note 1 above.
        S2I = pyro.sample("S2I_{}".format(t), dist.Binomial(S, prob_s * I / population),
                          obs=S2I)
        I2R = pyro.sample("I2R_{}".format(t), dist.Binomial(I, prob_i),
                          obs=I2R)
        S = S_next
        I = I_next
        # See Note 1 above.
        pyro.sample("obs_{}".format(t), dist.Binomial(I, rho),
                    obs=datum)


"""
The above model still does not allow HMC inference because the (S_aux,I_aux)
variables are discrete. However we have replaced dynamic integer_interval
constraints with easier static nonnegative_integer constraints (although we'll
still need good initialization to avoid NANs). More importantly, we have
converted to coordinates where the model is Markov.

Next we'll continue to reparameterize, this time replacing each of
(S_aux,I_aux) with a combination of a positive real variable and a Bernoulli
variable.

This is the crux: we can now perform HMC over the real variable and marginalize
out the Bernoulli using variable elimination.
"""


def quantize(name, x_real):
    """Randomly quantize in a way that preserves probability mass."""
    # Note this is a linear spline, but we could easily replace with a
    # quadratic spline to ensure gradients are continuous.
    lb = x_real.floor()
    bern = pyro.sample(name, dist.Bernoulli(x_real - lb))
    return lb + bern


@config_enumerate
def continuous_model(data):
    # Sample global parameters.
    prob_s = pyro.sample("prob_s", dist.Uniform(0, 1))
    prob_i = pyro.sample("prob_i", dist.Uniform(0, 1))
    rho = pyro.sample("rho", dist.Uniform(0, 1))
    population = 10000

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
        S2I = pyro.sample("S2I_{}".format(t), dist.Binomial(S, prob_s * I / population),
                          obs=S2I)
        I2R = pyro.sample("I2R_{}".format(t), dist.Binomial(I, prob_i),
                          obs=I2R)
        S = S_next
        I = I_next
        # See Note 1 above.
        pyro.sample("obs_{}".format(t), dist.Binomial(I, rho),
                    obs=datum)


"""
Tada :) All latent variables in the continuous_model are either continuous or
enumerated, so we can use HMC.

One potential enhancement would be to use DiscreteCosineReparam for the
S_aux, I_aux variables, improving the geometry of the inference problem.
That can easily be applied later, outside of the model.

The next step is to vectorize. We can repurpose DiscreteHMM here, but we'll
need to manually represent a Markov neighborhood of multiple Bernoullis as
single joint Categorical (I think just 4 states).
"""


# TODO
