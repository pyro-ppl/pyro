import pyro
import torch

from pyro.contrib.autoguide import AutoDelta
from pyro.infer import TraceEnum_ELBO, SVI
from pyro.infer.mcmc import NUTS, MCMC
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.util import ignore_jit_warnings


def hmm(num_steps, jit=True, init_trace=False):
    dim = 4

    def model(data):
        initialize = pyro.sample("initialize", dist.Dirichlet(torch.ones(dim)))
        with pyro.plate("states", dim):
            transition = pyro.sample("transition", dist.Dirichlet(torch.ones(dim, dim)))
            emission_loc = pyro.sample("emission_loc", dist.Normal(torch.zeros(dim), torch.ones(dim)))
            emission_scale = 1.
        x = None
        with ignore_jit_warnings([("Iterating over a tensor", RuntimeWarning)]):
            for t, y in pyro.markov(enumerate(data)):
                x = pyro.sample("x_{}".format(t),
                                dist.Categorical(initialize if x is None else transition[x]),
                                infer={"enumerate": "parallel"})
                pyro.sample("y_{}".format(t), dist.Normal(emission_loc[x], emission_scale), obs=y)

    def _get_initial_trace():
        guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: not msg["name"].startswith("x") and
                                        not msg["name"].startswith("y")))
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        svi = SVI(model, guide, optim.Adam({"lr": .01}), elbo, num_steps=100).run(data)
        return svi.exec_traces[-1]

    def _generate_data():
        transition_probs = torch.rand(dim, dim)
        emissions_loc = torch.arange(dim, dtype=torch.Tensor().dtype)
        emissions_scale = 1.
        state = torch.tensor(1)
        obs = [dist.Normal(emissions_loc[state], emissions_scale).sample()]
        for _ in range(num_steps):
            state = dist.Categorical(transition_probs[state]).sample()
            obs.append(dist.Normal(emissions_loc[state], emissions_scale).sample())
        return torch.stack(obs)

    data = _generate_data()
    nuts_kernel = NUTS(model, max_plate_nesting=1, jit_compile=jit, ignore_jit_warnings=True)
    if init_trace:
        nuts_kernel.initial_trace = _get_initial_trace()
    mcmc = MCMC(nuts_kernel, num_samples=300, warmup_steps=10).run(data)
    print(mcmc.marginal(sites=["emission_loc"]).empirical["emission_loc"].mean)


pyro.set_rng_seed(0)
hmm(200)
