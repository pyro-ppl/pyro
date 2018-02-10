from __future__ import absolute_import, division, print_function

import logging

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI
from tests.common import assert_equal

logger = logging.getLogger(__name__)


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.init(rng_seed=161)
@pytest.mark.parametrize("batch_size", [3, 5, 7, 8, None])
@pytest.mark.parametrize("map_type", ["tensor", "list"])
def test_elbo_mapdata(batch_size, map_type):
    # normal-normal: known covariance
    lam0 = Variable(torch.Tensor([0.1, 0.1]))   # precision of prior
    mu0 = Variable(torch.Tensor([0.0, 0.5]))   # prior mean
    # known precision of observation noise
    lam = Variable(torch.Tensor([6.0, 4.0]))
    data = []
    sum_data = Variable(torch.zeros(2))

    def add_data_point(x, y):
        data.append(Variable(torch.Tensor([x, y])))
        sum_data.data.add_(data[-1].data)

    add_data_point(0.1, 0.21)
    add_data_point(0.16, 0.11)
    add_data_point(0.06, 0.31)
    add_data_point(-0.01, 0.07)
    add_data_point(0.23, 0.25)
    add_data_point(0.19, 0.18)
    add_data_point(0.09, 0.41)
    add_data_point(-0.04, 0.17)

    n_data = Variable(torch.Tensor([len(data)]))
    analytic_lam_n = lam0 + n_data.expand_as(lam) * lam
    analytic_log_sig_n = -0.5 * torch.log(analytic_lam_n)
    analytic_mu_n = sum_data * (lam / analytic_lam_n) +\
        mu0 * (lam0 / analytic_lam_n)
    n_steps = 7000

    logger.debug("DOING ELBO TEST [bs = {}, map_type = {}]".format(batch_size, map_type))
    pyro.clear_param_store()

    def model():
        mu_latent = pyro.sample("mu_latent",
                                dist.Normal(mu0, torch.pow(lam0, -0.5)))
        if map_type == "list":
            pyro.map_data("aaa", data,
                          lambda i, x: pyro.sample("obs_%d" % i,
                                                   dist.Normal(mu_latent, torch.pow(lam, -0.5)),
                                                   obs=x),
                          batch_size=batch_size)
        elif map_type == "tensor":
            tdata = torch.cat([xi.view(1, -1) for xi in data], 0)
            pyro.map_data("aaa", tdata,
                          # XXX get batch size args to dist right
                          lambda i, x: pyro.sample("obs",
                                                   dist.Normal(mu_latent, torch.pow(lam, -0.5)),
                                                   obs=x),
                          batch_size=batch_size)
        else:
            for i, x in enumerate(data):
                pyro.observe('obs_%d' % i,
                             dist.Normal(mu_latent, torch.pow(lam, -0.5)),
                             obs=x)
        return mu_latent

    def guide():
        mu_q = pyro.param("mu_q", Variable(analytic_mu_n.data + torch.Tensor([-0.18, 0.23]),
                                           requires_grad=True))
        log_sig_q = pyro.param("log_sig_q", Variable(
            analytic_log_sig_n.data - torch.Tensor([-0.18, 0.23]),
            requires_grad=True))
        sig_q = torch.exp(log_sig_q)
        pyro.sample("mu_latent", dist.Normal(mu_q, sig_q))
        if map_type == "list" or map_type is None:
            pyro.map_data("aaa", data, lambda i, x: None, batch_size=batch_size)
        elif map_type == "tensor":
            tdata = torch.cat([xi.view(1, -1) for xi in data], 0)
            # dummy map_data to do subsampling for observe
            pyro.map_data("aaa", tdata, lambda i, x: None, batch_size=batch_size)
        else:
            pass

    adam = optim.Adam({"lr": 0.0008, "betas": (0.95, 0.999)})
    svi = SVI(model, guide, adam, loss="ELBO", trace_graph=True)

    for k in range(n_steps):
        svi.step()

        mu_error = torch.sum(
            torch.pow(
                analytic_mu_n -
                pyro.param("mu_q"),
                2.0))
        log_sig_error = torch.sum(
            torch.pow(
                analytic_log_sig_n -
                pyro.param("log_sig_q"),
                2.0))

        if k % 500 == 0:
            logger.debug("errors - {}, {}".format(mu_error.data.cpu().numpy()[0], log_sig_error.data.cpu().numpy()[0]))

    assert_equal(Variable(torch.zeros(1)), mu_error, prec=0.05)
    assert_equal(Variable(torch.zeros(1)), log_sig_error, prec=0.06)
