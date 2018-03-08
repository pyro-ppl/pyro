import argparse
import cProfile
import numbers
import time

import os
from collections import namedtuple

import pytest
import re
import six
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.testing import fakes
from pyro.infer import SVI
import pyro.optim as optim
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS


Model = namedtuple('TestModel', ['model', 'model_args'])


TEST_MODELS = []
MODEL_IDS = []


def model_id(test_model):
    def serialize(value):
        if isinstance(value, (numbers.Number, str)):
            return str(value)
        if hasattr(value, '__name__'):
            return value.__name__
        return None

    argstring = "_".join([str(k) + "=" + serialize(v) for k, v in test_model.model_args.items()
                          if serialize(v) is not None])
    return test_model.model.__name__ + "::" + argstring


def register_model(**model_kwargs):
    def register_fn(model):
        test_model = Model(model, model_kwargs)
        TEST_MODELS.append(test_model)
        MODEL_IDS.append(model_id(test_model))
        return model
    return register_fn


@register_model(reparameterized=True)
@register_model(reparameterized=False)
def poisson_gamma_model(reparameterized):
    alpha0 = torch.tensor(1.0)
    beta0 = torch.tensor(1.0)
    data = torch.tensor([1.0, 2.0, 3.0])
    n_data = len(data)
    data_sum = data.sum(0)
    alpha_n = alpha0 + data_sum  # posterior alpha
    beta_n = beta0 + torch.tensor(n_data)  # posterior beta
    log_alpha_n = torch.log(alpha_n)
    log_beta_n = torch.log(beta_n)

    pyro.clear_param_store()
    Gamma = dist.Gamma if reparameterized else fakes.NonreparameterizedGamma

    def model():
        lambda_latent = pyro.sample("lambda_latent", Gamma(alpha0, beta0))
        with pyro.iarange("data", n_data):
            pyro.sample("obs", dist.Poisson(lambda_latent), obs=data)
        return lambda_latent

    def guide():
        alpha_q_log = pyro.param(
            "alpha_q_log",
            torch.tensor(
                log_alpha_n.data +
                0.17,
                requires_grad=True))
        beta_q_log = pyro.param(
            "beta_q_log",
            torch.tensor(
                log_beta_n.data -
                0.143,
                requires_grad=True))
        alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
        pyro.sample("lambda_latent", Gamma(alpha_q, beta_q))

    adam = optim.Adam({"lr": .0002, "betas": (0.97, 0.999)})
    svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)
    for k in range(1000):
        svi.step()


@register_model(kernel=NUTS, step_size=0.02)
@register_model(kernel=HMC, step_size=0.02, num_steps=3)
def bernoulli_beta_hmc(**kwargs):
    def model(data):
        alpha = pyro.param('alpha', torch.tensor([1.1, 1.1], requires_grad=True))
        beta = pyro.param('beta', torch.tensor([1.1, 1.1], requires_grad=True))
        p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
        pyro.observe("obs", dist.Bernoulli(p_latent), data)
        return p_latent
    kernel = kwargs.pop('kernel')
    mcmc_kernel = kernel(model, **kwargs)
    mcmc_run = MCMC(mcmc_kernel, num_samples=300, warmup_steps=50)
    posterior = []
    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    for trace, _ in mcmc_run._traces(data):
        posterior.append(trace.nodes['p_latent']['value'])


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PROF_DIR = os.path.join(ROOT_DIR, ".benchmarks")
if not os.path.exists(PROF_DIR):
    os.makedirs(PROF_DIR)


@pytest.mark.parametrize('model, model_args', TEST_MODELS, ids=MODEL_IDS)
@pytest.mark.benchmark(
    min_rounds=5,
    disable_gc=True,
)
def test_benchmarks(benchmark, model, model_args):
    benchmark(model, **model_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiling different Pyro models.")
    parser.add_argument("-m", "--models", nargs="*")
    parser.add_argument("-b", "--suffix", default="current_branch")
    args = parser.parse_args()
    search_regexp = [re.compile(m) for m in args.models]
    model_names = [m.__name__ for m in MODELS]
    to_profile = []
    for r in search_regexp:
        to_profile += filter(r.match, model_names)
    to_profile = set(to_profile) if to_profile else model_names
    for model in to_profile:
        print("Running model - {}".format(model))
        pr = cProfile.Profile()
        pr.runctx(model + '()', globals(), locals())
        profile_file = os.path.join(PROF_DIR, model + "#" + args.suffix + ".prof")
        pr.dump_stats(profile_file)
        print("Results in - {}".format(profile_file))
