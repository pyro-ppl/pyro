import pyro
import torch
from pyro.infer.mcmc import NUTS, MCMC, HMC
import pyro.distributions as dist


def gaussian_hmm_enum_shape(num_steps):
    dim = 4

    def model(data):
        initialize = pyro.sample("initialize", dist.Dirichlet(torch.ones(dim)))
        transition = pyro.sample("transition", dist.Dirichlet(torch.ones(dim, dim)))
        emission_loc = pyro.sample("emission_loc", dist.Normal(torch.zeros(dim), torch.ones(dim)))
        emission_scale = pyro.sample("emission_scale", dist.LogNormal(torch.zeros(dim), torch.ones(dim)))
        x = None
        for t, y in enumerate(data):
            x = pyro.sample("x_{}".format(t), dist.Categorical(initialize if x is None else transition[x]))
            pyro.sample("y_{}".format(t), dist.Normal(emission_loc[x], emission_scale[x]), obs=y)
            # check shape
            effective_dim = sum(1 for size in x.shape if size > 1)
            assert effective_dim == 1

    data = torch.ones(num_steps)
    nuts_kernel = NUTS(model, max_plate_nesting=0, experimental_use_einsum=True,
                       full_mass=True, step_size=0.1, adapt_step_size=False)
    MCMC(nuts_kernel, num_samples=10, warmup_steps=10).run(data)


pyro.set_rng_seed(0)
gaussian_hmm_enum_shape(10)
