from __future__ import absolute_import, division, print_function

import argparse
import cProfile
import os
import re
from collections import namedtuple

import pytest
import torch

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.optim as optim
from pyro.distributions.testing import fakes
from pyro.infer import SVI, EmpiricalMarginal, Trace_ELBO, TraceGraph_ELBO
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.mcmc.nuts import NUTS

Model = namedtuple('TestModel', ['model', 'model_args', 'model_id'])


TEST_MODELS = []
MODEL_IDS = []
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
PROF_DIR = os.path.join(ROOT_DIR, ".benchmarks")
if not os.path.exists(PROF_DIR):
    os.makedirs(PROF_DIR)


def register_model(**model_kwargs):
    def register_fn(model):
        model_id = model_kwargs.pop("id")
        test_model = Model(model, model_kwargs, model_id)
        TEST_MODELS.append(test_model)
        MODEL_IDS.append(model_id)
        return model
    return register_fn


@register_model(reparameterized=True, Elbo=TraceGraph_ELBO, id='PoissonGamma::reparam=True_TraceGraph')
@register_model(reparameterized=True, Elbo=Trace_ELBO, id='PoissonGamma::reparam=True_Trace')
@register_model(reparameterized=False, Elbo=TraceGraph_ELBO, id='PoissonGamma::reparam=False_TraceGraph')
@register_model(reparameterized=False, Elbo=Trace_ELBO, id='PoissonGamma::reparam=False_Trace')
def poisson_gamma_model(reparameterized, Elbo):
    alpha0 = torch.tensor(1.0)
    beta0 = torch.tensor(1.0)
    data = torch.tensor([1.0, 2.0, 3.0])
    n_data = len(data)
    data_sum = data.sum(0)
    alpha_n = alpha0 + data_sum  # posterior alpha
    beta_n = beta0 + torch.tensor(float(n_data))  # posterior beta
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
    svi = SVI(model, guide, adam, loss=Elbo())
    for k in range(3000):
        svi.step()


@register_model(kernel=NUTS, step_size=0.02, num_samples=300, id='BernoulliBeta::NUTS')
@register_model(kernel=HMC, step_size=0.02, num_steps=3, num_samples=1000, id='BernoulliBeta::HMC')
def bernoulli_beta_hmc(**kwargs):
    def model(data):
        alpha = pyro.param('alpha', torch.tensor([1.1, 1.1]))
        beta = pyro.param('beta', torch.tensor([1.1, 1.1]))
        p_latent = pyro.sample("p_latent", dist.Beta(alpha, beta))
        pyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
        return p_latent

    true_probs = torch.tensor([0.9, 0.1])
    data = dist.Bernoulli(true_probs).sample(sample_shape=(torch.Size((1000,))))
    kernel = kwargs.pop('kernel')
    num_samples = kwargs.pop('num_samples')
    mcmc_kernel = kernel(model, **kwargs)
    mcmc_run = MCMC(mcmc_kernel, num_samples=num_samples, warmup_steps=100).run(data)
    return EmpiricalMarginal(mcmc_run, sites='p_latent')


@register_model(num_steps=2000, whiten=False, id='SVGP::MultiClass_whiten=False')
@register_model(num_steps=2000, whiten=True, id='SVGP::MultiClass_whiten=True')
def svgp_multiclass(num_steps, whiten):
    # adapted from http://gpflow.readthedocs.io/en/latest/notebooks/multiclass.html
    X = torch.rand(100, 1)
    K = (-0.5 * (X - X.t()).pow(2) / 0.01).exp() + torch.eye(100) * 1e-6
    f = K.potrf(upper=False).matmul(torch.randn(100, 3))
    y = f.argmax(dim=-1)

    kernel = gp.kernels.Matern32(1).add(
        gp.kernels.WhiteNoise(1, variance=torch.tensor(0.01)))
    likelihood = gp.likelihoods.MultiClass(num_classes=3)
    Xu = X[::5].clone()

    gpmodel = gp.models.VariationalSparseGP(X, y, kernel, Xu, likelihood,
                                            latent_shape=torch.Size([3]),
                                            whiten=whiten)

    gpmodel.fix_param("Xu")
    gpmodel.kernel.get_subkernel("WhiteNoise").fix_param("variance")

    gpmodel.optimize(optim.Adam({"lr": 0.0001}), num_steps=num_steps)


@pytest.mark.parametrize('model, model_args, id', TEST_MODELS, ids=MODEL_IDS)
@pytest.mark.benchmark(
    min_rounds=5,
    disable_gc=True,
)
@pytest.mark.disable_validation()
def test_benchmark(benchmark, model, model_args, id):
    print("Running - {}".format(id))
    benchmark(model, **model_args)


def profile_fn(test_model):
    def wrapped():
        test_model.model(**test_model.model_args)
    return wrapped


if __name__ == "__main__":
    """
    This script is invoked to run cProfile on one of the models specified above.
    """
    parser = argparse.ArgumentParser(description="Profiling different Pyro models.")
    parser.add_argument("-m", "--models", nargs="*",
                        help="model name to match against model id, partial match (e.g. *NAME*) is acceptable.")
    parser.add_argument("-b", "--suffix", default="current_branch",
                        help="suffix to append to the cprofile output dump.")
    parser.add_argument("-d", "--benchmark_dir", default=PROF_DIR,
                        help="directory to save profiling benchmarks.")
    args = parser.parse_args()
    search_regexp = [re.compile(".*" + m + ".*") for m in args.models]
    profile_ids = []
    for r in search_regexp:
        profile_ids.append(filter(r.match, MODEL_IDS))
    profile_ids = set().union(*profile_ids)
    to_profile = [m for m in TEST_MODELS if m.model_id in profile_ids]
    # run cProfile for all models if not specified
    if not args.models:
        to_profile = TEST_MODELS
    for test_model in to_profile:
        print("Running model - {}".format(test_model.model_id))
        pr = cProfile.Profile()
        fn = profile_fn(test_model)
        pr.runctx("fn()", globals(), locals())
        profile_file = os.path.join(args.benchmark_dir, test_model.model_id + "#" + args.suffix + ".prof")
        pr.dump_stats(profile_file)
        print("Results in - {}".format(profile_file))
