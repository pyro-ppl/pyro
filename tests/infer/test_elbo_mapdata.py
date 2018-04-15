from __future__ import absolute_import, division, print_function

import logging

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
from tests.common import assert_equal

logger = logging.getLogger(__name__)


@pytest.mark.stage("integration", "integration_batch_1")
@pytest.mark.init(rng_seed=161)
@pytest.mark.parametrize("batch_size", [3, 8, None])
@pytest.mark.parametrize("map_type", ["iarange", "irange", "range"])
def test_elbo_mapdata(batch_size, map_type):
    # normal-normal: known covariance
    lam0 = torch.tensor([0.1, 0.1])   # precision of prior
    loc0 = torch.tensor([0.0, 0.5])   # prior mean
    # known precision of observation noise
    lam = torch.tensor([6.0, 4.0])
    data = []
    sum_data = torch.zeros(2)

    def add_data_point(x, y):
        data.append(torch.tensor([x, y]))
        sum_data.data.add_(data[-1].data)

    add_data_point(0.1, 0.21)
    add_data_point(0.16, 0.11)
    add_data_point(0.06, 0.31)
    add_data_point(-0.01, 0.07)
    add_data_point(0.23, 0.25)
    add_data_point(0.19, 0.18)
    add_data_point(0.09, 0.41)
    add_data_point(-0.04, 0.17)

    data = torch.stack(data)
    n_data = torch.tensor([float(len(data))])
    analytic_lam_n = lam0 + n_data.expand_as(lam) * lam
    analytic_log_sig_n = -0.5 * torch.log(analytic_lam_n)
    analytic_loc_n = sum_data * (lam / analytic_lam_n) +\
        loc0 * (lam0 / analytic_lam_n)
    n_steps = 7000

    logger.debug("DOING ELBO TEST [bs = {}, map_type = {}]".format(batch_size, map_type))
    pyro.clear_param_store()

    def model():
        loc_latent = pyro.sample("loc_latent",
                                 dist.Normal(loc0, torch.pow(lam0, -0.5)).independent(1))
        if map_type == "irange":
            for i in pyro.irange("aaa", len(data), batch_size):
                pyro.sample("obs_%d" % i, dist.Normal(loc_latent, torch.pow(lam, -0.5)) .independent(1),
                            obs=data[i]),
        elif map_type == "iarange":
            with pyro.iarange("aaa", len(data), batch_size) as ind:
                pyro.sample("obs", dist.Normal(loc_latent, torch.pow(lam, -0.5)) .independent(1),
                            obs=data[ind]),
        else:
            for i, x in enumerate(data):
                pyro.sample('obs_%d' % i,
                            dist.Normal(loc_latent, torch.pow(lam, -0.5))
                            .independent(1),
                            obs=x)
        return loc_latent

    def guide():
        loc_q = pyro.param("loc_q", torch.tensor(
            analytic_loc_n.data + torch.tensor([-0.18, 0.23]), requires_grad=True))
        log_sig_q = pyro.param("log_sig_q", torch.tensor(
            analytic_log_sig_n.data - torch.tensor([-0.18, 0.23]), requires_grad=True))
        sig_q = torch.exp(log_sig_q)
        pyro.sample("loc_latent", dist.Normal(loc_q, sig_q).independent(1))
        if map_type == "irange" or map_type is None:
            for i in pyro.irange("aaa", len(data), batch_size):
                pass
        elif map_type == "iarange":
            # dummy iarange to do subsampling for observe
            with pyro.iarange("aaa", len(data), batch_size):
                pass
        else:
            pass

    adam = optim.Adam({"lr": 0.0008, "betas": (0.95, 0.999)})
    svi = SVI(model, guide, adam, loss=TraceGraph_ELBO())

    for k in range(n_steps):
        svi.step()

        loc_error = torch.sum(
            torch.pow(
                analytic_loc_n -
                pyro.param("loc_q"),
                2.0))
        log_sig_error = torch.sum(
            torch.pow(
                analytic_log_sig_n -
                pyro.param("log_sig_q"),
                2.0))

        if k % 500 == 0:
            logger.debug("errors - {}, {}".format(loc_error, log_sig_error))

    assert_equal(loc_error.item(), 0, prec=0.05)
    assert_equal(log_sig_error.item(), 0, prec=0.06)
