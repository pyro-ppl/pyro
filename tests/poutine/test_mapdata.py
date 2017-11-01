from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer import SVI
from tests.common import assert_equal, requires_cuda


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
    verbose = True
    n_steps = 7000

    if verbose:
        print("DOING ELBO TEST [bs = {}, map_type = {}]".format(
            batch_size, map_type))
    pyro.clear_param_store()

    def model():
        mu_latent = pyro.sample("mu_latent", dist.normal,
                                mu0, torch.pow(lam0, -0.5))
        if map_type == "list":
            pyro.map_data("aaa", data, lambda i,
                          x: pyro.observe(
                              "obs_%d" % i, dist.normal,
                              x, mu_latent, torch.pow(lam, -0.5)), batch_size=batch_size)
        elif map_type == "tensor":
            tdata = torch.cat([xi.view(1, -1) for xi in data], 0)
            pyro.map_data("aaa", tdata,
                          # XXX get batch size args to dist right
                          lambda i, x: pyro.observe("obs", dist.normal, x, mu_latent,
                                                    torch.pow(lam, -0.5)),
                          batch_size=batch_size)
        else:
            for i, x in enumerate(data):
                pyro.observe('obs_%d' % i,
                             dist.normal, x, mu_latent, torch.pow(lam, -0.5))
        return mu_latent

    def guide():
        mu_q = pyro.param("mu_q", Variable(analytic_mu_n.data + torch.Tensor([-0.18, 0.23]),
                                           requires_grad=True))
        log_sig_q = pyro.param("log_sig_q", Variable(
            analytic_log_sig_n.data - torch.Tensor([-0.18, 0.23]),
            requires_grad=True))
        sig_q = torch.exp(log_sig_q)
        pyro.sample("mu_latent", dist.normal, mu_q, sig_q)
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
            print("errors", mu_error.data.cpu().numpy()[0], log_sig_error.data.cpu().numpy()[0])

    assert_equal(Variable(torch.zeros(1)), mu_error, prec=0.05)
    assert_equal(Variable(torch.zeros(1)), log_sig_error, prec=0.06)


@pytest.mark.parametrize("batch_dim", [0, 1])
def test_batch_dim(batch_dim):

    data = Variable(torch.randn(4, 5, 7))

    def local_model(ixs, _xs):
        xs = _xs.view(-1, _xs.size(2))
        return pyro.sample("xs", dist.normal,
                           xs, Variable(torch.ones(xs.size())))

    def model():
        return pyro.map_data("md", data, local_model,
                             batch_size=1, batch_dim=batch_dim)

    tr = poutine.trace(model).get_trace()
    assert tr.nodes["xs"]["value"].size(0) == data.size(1 - batch_dim)
    assert tr.nodes["xs"]["value"].size(1) == data.size(2)


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
                                                       dist.normal, x, y),
                                           batch_size=std_batch_size),
                             batch_size=mean_batch_size)

    model = model

    xs = model(means, stds)
    assert len(xs) == mean_batch_size
    assert len(xs[0]) == std_batch_size

    tr = poutine.trace(model).get_trace(means, stds)
    for name in tr.nodes.keys():
        if tr.nodes[name]["type"] == "sample" and name.startswith("x_"):
            assert tr.nodes[name]["scale"] == 4.0 * 2.0


def iarange_model(subsample_size):
    mu = Variable(torch.zeros(20))
    sigma = Variable(torch.ones(20))
    with pyro.iarange('iarange', 20, subsample_size) as batch:
        pyro.sample("x", dist.normal, mu[batch], sigma[batch])
        result = list(batch.data)
    return result


def irange_model(subsample_size):
    mu = Variable(torch.zeros(20))
    sigma = Variable(torch.ones(20))
    result = []
    for i in pyro.irange('irange', 20, subsample_size):
        pyro.sample("x_{}".format(i), dist.normal, mu[i], sigma[i])
        result.append(i)
    return result


def nested_irange_model(subsample_size):
    mu = Variable(torch.zeros(20))
    sigma = Variable(torch.ones(20))
    result = []
    for i in pyro.irange("outer", 20, subsample_size):
        result.append([])
        for j in pyro.irange("inner", 20, 5):
            pyro.sample("x_{}_{}".format(i, j), dist.normal, mu[i] + mu[j], sigma[i] + sigma[j])
            result[-1].append(j)
    return result


def map_data_vector_model(subsample_size):
    mu = Variable(torch.zeros(20))
    sigma = Variable(torch.ones(20))

    def local_model(batch, unused):
        pyro.sample("x", dist.normal, mu[batch], sigma[batch])
        return batch

    LongTensor = torch.cuda.LongTensor if torch.Tensor.is_cuda else torch.LongTensor
    ind = Variable(LongTensor(range(20)))
    batch = pyro.map_data('mapdata', ind, local_model, batch_size=subsample_size)
    return list(batch.data)


def map_data_iter_model(subsample_size):
    mu = Variable(torch.zeros(20))
    sigma = Variable(torch.ones(20))

    def local_model(i, unused):
        pyro.sample("x_{}".format(i), dist.normal, mu[i], sigma[i])
        return i

    return pyro.map_data('mapdata', range(20), local_model, batch_size=subsample_size)


@pytest.mark.parametrize('subsample_size', [5, 20])
@pytest.mark.parametrize('model', [
    iarange_model,
    irange_model,
    nested_irange_model,
    map_data_vector_model,
    map_data_iter_model,
], ids=['iarange', 'irange', 'nested_irange', 'map_data_vector', 'map_data_iter'])
def test_map_data_stack(model, subsample_size):
    tr = poutine.trace(model).get_trace(subsample_size)
    for name, node in tr.nodes.items():
        if name.startswith("x"):
            assert node["map_data_stack"], "missing map_data_stack at node {}".format(name)


@pytest.mark.parametrize('subsample_size', [5, 20])
@pytest.mark.parametrize('model', [
    iarange_model,
    irange_model,
    nested_irange_model,
    map_data_vector_model,
    map_data_iter_model,
], ids=['iarange', 'irange', 'nested_irange', 'map_data_vector', 'map_data_iter'])
def test_replay(model, subsample_size):
    pyro.set_rng_seed(0)

    traced_model = poutine.trace(model)
    original = traced_model(subsample_size)

    replayed = poutine.replay(model, traced_model.trace)(subsample_size)
    assert replayed == original

    if subsample_size < 20:
        different = traced_model(subsample_size)
        assert different != original


def iarange_custom_model(subsample):
    with pyro.iarange('iarange', 20, subsample=subsample) as batch:
        result = batch
    return result


def irange_custom_model(subsample):
    result = []
    for i in pyro.irange('irange', 20, subsample=subsample):
        result.append(i)
    return result


@pytest.mark.parametrize('model', [iarange_custom_model, irange_custom_model],
                         ids=['iarange', 'irange'])
def test_custom_subsample(model):
    pyro.set_rng_seed(0)

    subsample = [1, 3, 5, 7]
    assert model(subsample) == subsample
    assert poutine.trace(model)(subsample) == subsample


def iarange_cuda_model(subsample_size):
    mu = Variable(torch.zeros(20).cuda())
    sigma = Variable(torch.ones(20).cuda())
    with pyro.iarange("data", 20, subsample_size, use_cuda=True) as batch:
        pyro.sample("x", dist.normal, mu[batch], sigma[batch])


def irange_cuda_model(subsample_size):
    mu = Variable(torch.zeros(20).cuda())
    sigma = Variable(torch.ones(20).cuda())
    for i in pyro.irange("data", 20, subsample_size, use_cuda=True):
        pyro.sample("x_{}".format(i), dist.normal, mu[i], sigma[i])


def map_data_vector_cuda_model(subsample_size):
    mu = Variable(torch.zeros(20).cuda())
    sigma = Variable(torch.ones(20).cuda())
    pyro.map_data("data", mu,
                  lambda i, mu: pyro.sample("x", dist.normal, mu, sigma[i]),
                  batch_size=subsample_size, use_cuda=True)


def map_data_iter_cuda_model(subsample_size):
    mu = Variable(torch.zeros(20).cuda())
    sigma = Variable(torch.ones(20).cuda())
    pyro.map_data("data", list(mu),
                  lambda i, mu: pyro.sample("x_{}".format(i), dist.normal, mu, sigma[i]),
                  batch_size=subsample_size, use_cuda=True)


@requires_cuda
@pytest.mark.parametrize('subsample_size', [5, 20])
@pytest.mark.parametrize('model', [
    iarange_cuda_model,
    irange_cuda_model,
    map_data_vector_cuda_model,
    map_data_iter_cuda_model,
], ids=["iarange", "irange", "map_data_vector", "map_data_iter"])
def test_cuda(model, subsample_size):
    tr = poutine.trace(model).get_trace(subsample_size)
    assert tr.log_pdf().is_cuda
    assert tr.batch_log_pdf().is_cuda


@pytest.mark.parametrize('model', [
    iarange_model,
    irange_model,
    map_data_vector_model,
    map_data_iter_model,
], ids=['iarange', 'irange', 'map_data_vector', 'map_data_iter'])
@pytest.mark.parametrize("behavior,model_size,guide_size", [
    ("error", 20, 5),
    ("error", 5, 20),
    ("error", 5, None),
    ("ok", 20, 20),
    ("ok", 20, None),
    ("ok", 5, 5),
    ("ok", None, 20),
    ("ok", None, 5),
    ("ok", None, None),
])
def test_model_guide_mismatch(behavior, model_size, guide_size, model):
    model = poutine.trace(model)
    expected_ind = model(guide_size)
    if behavior == "ok":
        actual_ind = poutine.replay(model, model.trace)(model_size)
        assert actual_ind == expected_ind
    else:
        with pytest.raises(ValueError):
            poutine.replay(model, model.trace)(model_size)
