import torch
from torch.autograd import Variable
import pytest

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import pyro.optim as optim
from pyro.infer import SVI
from tests.common import assert_equal

pytestmark = pytest.mark.stage("integration", "integration_batch_1")


@pytest.mark.init(rng_seed=161)
@pytest.mark.parametrize("batch_size", [3, 5, 7, 8, 0])
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
    verbose = True
    n_steps = 7000

    if verbose:
        print("DOING ELBO TEST [bs = {}, map_type = {}]".format(
            batch_size, map_type))
    pyro.get_param_store().clear()

    def model():
        mu_latent = pyro.sample("mu_latent", dist.diagnormal,
                                mu0, torch.pow(lam0, -0.5))
        if map_type == "list":
            pyro.map_data("aaa", data, lambda i,
                          x: pyro.observe(
                              "obs_%d" % i, dist.diagnormal,
                              x, mu_latent, torch.pow(lam, -0.5)), batch_size=batch_size)
        elif map_type == "tensor":
            tdata = torch.cat([xi.view(1, -1) for xi in data], 0)
            pyro.map_data("aaa", tdata,
                          # XXX get batch size args to dist right
                          lambda i, x: pyro.observe("obs", dist.diagnormal, x, mu_latent,
                                                    torch.pow(lam, -0.5)),
                          batch_size=batch_size)
        else:
            for i, x in enumerate(data):
                pyro.observe('obs_%d' % i,
                             dist.diagnormal, x, mu_latent, torch.pow(lam, -0.5))
        return mu_latent

    def guide():
        mu_q = pyro.param("mu_q", Variable(analytic_mu_n.data + torch.Tensor([-0.18, 0.23]),
                                           requires_grad=True))
        log_sig_q = pyro.param("log_sig_q", Variable(
            analytic_log_sig_n.data - torch.Tensor([-0.18, 0.23]),
            requires_grad=True))
        sig_q = torch.exp(log_sig_q)
        pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q)
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

        if verbose and k % 500 == 0:
            print("errors", mu_error.data.numpy()[0], log_sig_error.data.numpy()[0])

    assert_equal(Variable(torch.zeros(1)), mu_error, prec=0.05)
    assert_equal(Variable(torch.zeros(1)), log_sig_error, prec=0.06)


@pytest.mark.parametrize("batch_dim", [0, 1])
def test_batch_dim(batch_dim):

    data = Variable(torch.randn(4, 5, 7))

    def local_model(ixs, _xs):
        xs = _xs.view(-1, _xs.size(2))
        return pyro.sample("xs", dist.diagnormal,
                           xs, Variable(torch.ones(xs.size())))

    def model():
        return pyro.map_data("md", data, local_model,
                             batch_size=1, batch_dim=batch_dim)

    tr = poutine.trace(model).get_trace()
    assert tr["xs"]["value"].size(0) == data.size(1 - batch_dim)
    assert tr["xs"]["value"].size(1) == data.size(2)


def test_nested_map_data():
    means = [Variable(torch.randn(2)) for i in range(8)]
    mean_batch_size = 2
    stds = [Variable(torch.abs(torch.randn(2))) for i in range(6)]
    std_batch_size = 3

    def model(means, stds):
        return pyro.map_data("a", means,
                             lambda i, x:
                             pyro.map_data("a_{}".format(i), stds,
                                           lambda j, y:
                                           pyro.sample("x_{}{}".format(i, j),
                                                       dist.diagnormal, x, y),
                                           batch_size=std_batch_size),
                             batch_size=mean_batch_size)

    model = model

    xs = model(means, stds)
    assert len(xs) == mean_batch_size
    assert len(xs[0]) == std_batch_size

    tr = poutine.trace(model).get_trace(means, stds)
    for name in tr.keys():
        if tr[name]["type"] == "sample":
            assert tr[name]["scale"] == 4.0 * 2.0
