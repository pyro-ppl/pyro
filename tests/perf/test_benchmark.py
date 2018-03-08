import argparse
import cProfile
import time

import os
import pytest
import re
import six
import torch
from torch.autograd import variable

import pyro
import pyro.distributions as dist
from pyro.distributions.testing import fakes
from pyro.infer import SVI
import pyro.optim as optim

TIMER = time.clock if six.PY2 else time.process_time


def poisson_gamma_model():
    alpha0 = variable(1.0)
    beta0 = variable(1.0)
    data = variable([1.0, 2.0, 3.0])
    n_data = len(data)
    data_sum = data.sum(0)
    alpha_n = alpha0 + data_sum  # posterior alpha
    beta_n = beta0 + variable(n_data)  # posterior beta
    log_alpha_n = torch.log(alpha_n)
    log_beta_n = torch.log(beta_n)

    pyro.clear_param_store()
    Gamma = fakes.NonreparameterizedGamma

    def model():
        lambda_latent = pyro.sample("lambda_latent", Gamma(alpha0, beta0))
        with pyro.iarange("data", n_data):
            pyro.sample("obs", dist.Poisson(lambda_latent), obs=data)
        return lambda_latent

    def guide():
        alpha_q_log = pyro.param(
            "alpha_q_log",
            variable(
                log_alpha_n.data +
                0.17,
                requires_grad=True))
        beta_q_log = pyro.param(
            "beta_q_log",
            variable(
                log_beta_n.data -
                0.143,
                requires_grad=True))
        alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
        pyro.sample("lambda_latent", Gamma(alpha_q, beta_q))

    adam = optim.Adam({"lr": .0002, "betas": (0.97, 0.999)})
    svi = SVI(model, guide, adam, loss="ELBO", trace_graph=False)
    for k in range(1000):
        svi.step()


MODELS = [poisson_gamma_model]
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PROF_DIR = os.path.join(ROOT_DIR, ".benchmarks")
if not os.path.exists(PROF_DIR):
    os.makedirs(PROF_DIR)


@pytest.mark.benchmark(
    min_rounds=5,
    timer=TIMER,
    disable_gc=True,
)
def test_poisson_gamma(benchmark):
    benchmark(poisson_gamma_model)


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
