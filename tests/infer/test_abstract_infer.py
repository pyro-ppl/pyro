import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoLaplaceApproximation
from pyro.infer import SVI, TracePredictive, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS
from tests.common import assert_close, assert_equal


pytestmark = pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")


def model(num_trials):
    with pyro.plate("data", num_trials.size(0)):
        phi_prior = dist.Uniform(num_trials.new_tensor(0.), num_trials.new_tensor(1.))
        success_prob = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", dist.Binomial(num_trials, success_prob))


def one_hot_model(pseudocounts, classes=None):
    probs_prior = dist.Dirichlet(pseudocounts)
    probs = pyro.sample("probs", probs_prior)
    with pyro.plate("classes", classes.size(0) if classes is not None else 1, dim=-1):
        return pyro.sample("obs", dist.OneHotCategorical(probs), obs=classes)


def beta_guide(num_trials):
    phi_c0 = pyro.param("phi_c0", num_trials.new_tensor(5.0).expand([num_trials.size(0)]))
    phi_c1 = pyro.param("phi_c1", num_trials.new_tensor(5.0).expand([num_trials.size(0)]))
    with pyro.plate("data", num_trials.size(0)):
        phi_posterior = dist.Beta(concentration0=phi_c0, concentration1=phi_c1)
        pyro.sample("phi", phi_posterior)


def test_posterior_predictive_svi_manual_guide():
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    opt = optim.Adam(dict(lr=1.0))
    loss = Trace_ELBO()
    guide = beta_guide
    svi_run = SVI(conditioned_model, guide, opt, loss, num_steps=1000, num_samples=100).run(num_trials)
    posterior_predictive = TracePredictive(model, svi_run, num_samples=10000).run(num_trials[:3])
    marginal_return_vals = posterior_predictive.marginal().empirical["_RETURN"]
    assert_close(marginal_return_vals.mean, torch.ones(3) * 700, rtol=0.05)


def test_posterior_predictive_svi_auto_delta_guide():
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    opt = optim.Adam(dict(lr=1.0))
    loss = Trace_ELBO()
    guide = AutoDelta(conditioned_model)
    svi_run = SVI(conditioned_model, guide, opt, loss, num_steps=1000, num_samples=100).run(num_trials)
    posterior_predictive = TracePredictive(model, svi_run, num_samples=10000).run(num_trials)
    marginal_return_vals = posterior_predictive.marginal().empirical["_RETURN"]
    assert_close(marginal_return_vals.mean, torch.ones(5) * 700, rtol=0.05)


def test_posterior_predictive_svi_auto_diag_normal_guide():
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials, true_probs).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    opt = optim.Adam(dict(lr=0.1))
    loss = Trace_ELBO()
    guide = AutoDiagonalNormal(conditioned_model)
    svi_run = SVI(conditioned_model, guide, opt, loss, num_steps=1000, num_samples=100).run(num_trials)
    posterior_predictive = TracePredictive(model, svi_run, num_samples=10000).run(num_trials)
    marginal_return_vals = posterior_predictive.marginal().empirical["_RETURN"]
    assert_close(marginal_return_vals.mean, torch.ones(5) * 700, rtol=0.05)


def test_posterior_predictive_svi_one_hot():
    pseudocounts = torch.ones(3) * 0.1
    true_probs = torch.tensor([0.15, 0.6, 0.25])
    classes = dist.OneHotCategorical(true_probs).sample((10000,))
    opt = optim.Adam(dict(lr=0.1))
    loss = Trace_ELBO()
    guide = AutoDelta(one_hot_model)
    svi_run = SVI(one_hot_model, guide, opt, loss, num_steps=1000, num_samples=1000).run(pseudocounts, classes=classes)
    posterior_predictive = TracePredictive(one_hot_model, svi_run, num_samples=10000).run(pseudocounts)
    marginal_return_vals = posterior_predictive.marginal().empirical["_RETURN"]
    assert_close(marginal_return_vals.mean, true_probs.unsqueeze(0), rtol=0.1)


def test_posterior_predictive_svi_auto_delta_guide_large_eval():
    true_probs = torch.ones(5) * 0.7
    num_trials = torch.ones(5) * 1000
    num_success = dist.Binomial(num_trials[:3], true_probs[:3]).sample()
    conditioned_model = poutine.condition(model, data={"obs": num_success})
    opt = optim.Adam(dict(lr=1.0))
    loss = Trace_ELBO()
    guide = AutoDelta(conditioned_model)
    svi_run = SVI(conditioned_model, guide, opt, loss, num_steps=1000, num_samples=100).run(num_trials[:3])
    posterior_predictive = TracePredictive(model, svi_run, num_samples=10000).run(num_trials)
    marginal_return_vals = posterior_predictive.marginal().empirical["_RETURN"]
    assert_close(marginal_return_vals.mean, torch.ones(5) * 700, rtol=0.05)


def test_nesting():
    def nested():
        true_probs = torch.ones(5) * 0.7
        num_trials = torch.ones(5) * 1000
        num_success = dist.Binomial(num_trials, true_probs).sample()
        conditioned_model = poutine.condition(model, data={"obs": num_success})
        nuts_kernel = NUTS(conditioned_model, adapt_step_size=True)
        mcmc_run = MCMC(nuts_kernel, num_samples=10, warmup_steps=2).run(num_trials)
        return mcmc_run

    with poutine.trace() as tp:
        nested()
        nested()

    assert len(tp.trace.nodes) == 0


def test_information_criterion():
    # milk dataset: https://github.com/rmcelreath/rethinking/blob/master/data/milk.csv
    kcal = torch.tensor([0.49, 0.47, 0.56, 0.89, 0.92, 0.8, 0.46, 0.71, 0.68,
                         0.97, 0.84, 0.62, 0.54, 0.49, 0.48, 0.55, 0.71])
    kcal_mean = kcal.mean()
    kcal_logstd = kcal.std().log()

    def model():
        mu = pyro.sample("mu", dist.Normal(kcal_mean, 1))
        log_sigma = pyro.sample("log_sigma", dist.Normal(kcal_logstd, 1))
        with pyro.plate("plate"):
            pyro.sample("kcal", dist.Normal(mu, log_sigma.exp()), obs=kcal)

    delta_guide = AutoLaplaceApproximation(model)

    svi = SVI(model, delta_guide, optim.Adam({"lr": 0.05}), loss=Trace_ELBO(), num_samples=3000)
    for i in range(100):
        svi.step()

    svi.guide = delta_guide.laplace_approximation()
    posterior = svi.run()

    ic = posterior.information_criterion()
    assert_equal(ic["waic"], torch.tensor(-8.3), prec=0.2)
    assert_equal(ic["p_waic"], torch.tensor(1.8), prec=0.2)
