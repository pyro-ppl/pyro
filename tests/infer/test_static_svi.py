from __future__ import absolute_import, division, print_function

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import StaticSVI, Trace_ELBO

from tests.common import assert_equal


def test_inference():
    alpha0 = torch.tensor(1.0)
    beta0 = torch.tensor(1.0)  # beta prior hyperparameter
    data = torch.tensor([0.0, 1.0, 1.0, 1.0])
    n_data = len(data)
    data_sum = data.sum()
    alpha_n = alpha0 + data_sum  # posterior alpha
    beta_n = beta0 - data_sum + torch.tensor(float(n_data))  # posterior beta
    log_alpha_n = torch.log(alpha_n)
    log_beta_n = torch.log(beta_n)

    def model():
        p_latent = pyro.sample("p_latent", dist.Beta(alpha0, beta0))
        with pyro.plate("data", n_data):
            pyro.sample("obs", dist.Bernoulli(p_latent), obs=data)
        return p_latent

    def guide():
        alpha_q_log = pyro.param("alpha_q_log", log_alpha_n + 0.17)
        beta_q_log = pyro.param("beta_q_log", log_beta_n - 0.143)
        alpha_q, beta_q = torch.exp(alpha_q_log), torch.exp(beta_q_log)
        pyro.sample("p_latent", dist.Beta(alpha_q, beta_q))

    adam = optim.Adam({"lr": .001})
    svi = StaticSVI(model, guide, adam, loss=Trace_ELBO())
    for i in range(1000):
        svi.step()

    assert_equal(pyro.param("alpha_q_log"), log_alpha_n, prec=0.04)
    assert_equal(pyro.param("beta_q_log"), log_beta_n, prec=0.04)


def test_params_not_match():
    def model(i):
        p = pyro.param(str(i), torch.tensor(float(i)))
        return pyro.sample("obs", dist.Normal(p, 1), obs=torch.tensor(0.))

    def guide(i):
        pass

    adam = optim.Adam({})
    svi = StaticSVI(model, guide, adam, loss=Trace_ELBO())
    svi.step(i=0)

    with pytest.raises(ValueError, match="Param `{}` is not available".format(1)):
        svi.step(i=1)
