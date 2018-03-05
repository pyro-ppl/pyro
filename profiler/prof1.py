import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS
from tests.common import assert_equal


def model(data):
    alpha = pyro.param('alpha', Variable(torch.Tensor([1.1, 1.1]), requires_grad=True))
    beta = pyro.param('beta', Variable(torch.Tensor([1.1, 1.1]), requires_grad=True))
    p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
    pyro.observe("obs", dist.Bernoulli(p_latent), data)
    return p_latent


hmc_kernel = HMC(model, step_size=0.01, num_steps=3)
mcmc_run = MCMC(hmc_kernel, num_samples=40, warmup_steps=0)
posterior = []
true_probs = Variable(torch.Tensor([0.9, 0.1]))
data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
i = 0
for trace, _ in mcmc_run._traces(data):
    print(i)
    posterior.append(trace.nodes['p_latent']['value'])
    i += 1
