from collections import OrderedDict

import torch

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model, select_samples
from pyro.infer.mcmc.api import MCMC


def normal_normal_model():
    y = pyro.sample("y", dist.Normal(0.0, torch.tensor([1.0])))
    pyro.sample("obs", dist.Normal(y, 1.0), obs=torch.tensor([0.0]))


class RandomWalkKernel(MCMCKernel):
    r"""
    Simple gradient-free kernel that utilizes an isotropic random walk in the unconstrained
    space of the model.

    :param model: Python callable containing Pyro primitives.

    Example:

        >>> true_coefs = torch.tensor([1., 2., 3.])
        >>> data = torch.randn(2000, 3)
        >>> dim = 3
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
        >>>
        >>> def model(data):
        ...     coefs_mean = torch.zeros(dim)
        ...     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
        ...     y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        ...     return y
        >>>
        >>> hmc_kernel = RandomWalkKernel(model, step_size=0.01)
        >>> mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
        >>> mcmc.run(data)
        >>> mcmc.get_samples()['beta'].mean(0)  # doctest: +SKIP
        tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(self, model, step_size=0.3):
        self.model = model
        self.step_size = step_size

        self._t = 0
        self._accept_cnt = 0
        self._mean_accept_prob = 0.0

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        self._initial_params, self.potential_fn, self.transforms, self._prototype_trace = initialize_model(
            self.model, model_args=args, model_kwargs=kwargs,
        )
        self._energy_last = self.potential_fn(self._initial_params)

    def sample(self, params):
        new_params = {k: v + self.step_size * torch.randn(v.shape, dtype=v.dtype, device=v.device) for k, v in params.items()}
        energy_proposal = self.potential_fn(new_params)
        delta_energy = energy_proposal - self._energy_last

        accept_prob = (-delta_energy).exp().clamp(max=1.0)
        rand = pyro.sample(
            "rand_t={}".format(self._t),
            dist.Uniform(0.0, 1.0),
        )
        accepted = False
        if rand < accept_prob:
            accepted = True
            params = new_params
            self._energy_last = energy_proposal

        self._t += 1

        if self._t > self._warmup_steps:
            n = self._t - self._warmup_steps
            if accepted:
                self._accept_cnt += 1
        else:
            n = self._t

        self._mean_accept_prob += (accept_prob.item() - self._mean_accept_prob) / n

        return params.copy()

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def logging(self):
        return OrderedDict(
            [
                ("acc. prob", "{:.3f}".format(self._mean_accept_prob)),
            ]
        )

    def diagnostics(self):
        return {
            "acceptance rate": self._accept_cnt / (self._t - self._warmup_steps),
        }

kernel = RandomWalkKernel(normal_normal_model)

initial_params, _, transforms, _ = initialize_model(
    normal_normal_model, model_args=(),
)

mcmc = MCMC(
        kernel=kernel,
        num_samples=10000,
        warmup_steps=1000,
        initial_params=initial_params,
        num_chains=1,
        transforms=transforms,
    )

mcmc.run()

samples = mcmc.get_samples()
y = samples['y']
print("y: ", y.mean(), " +- ", y.std())
