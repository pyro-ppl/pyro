# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from scipy.special import binom
from torch import tensor
from torch.distributions import Beta, HalfNormal, VonMises

import pyro
from pyro.distributions import Geometric, constraints
from pyro.distributions.sine_bivariate_von_mises import SineBivariateVonMises
from pyro.infer import SVI, Trace_ELBO
from tests.common import assert_equal


def _unnorm_log_prob(value, loc1, loc2, conc1, conc2, corr):
    phi_val = value[..., 0]
    psi_val = value[..., 1]
    return (
        conc1 * torch.cos(phi_val - loc1)
        + conc2 * torch.cos(psi_val - loc2)
        + corr * torch.sin(phi_val - loc1) * torch.sin(psi_val - loc2)
    )


@pytest.mark.parametrize("n", [0, 1, 10, 20])
def test_log_binomial(n):
    comp = SineBivariateVonMises._lbinoms(tensor(n))
    act = tensor([binom(2 * i, i) for i in range(n)]).log()
    assert_equal(act, comp)


@pytest.mark.parametrize("batch_dim", [tuple(), (1,), (10,), (2, 1), (2, 1, 2)])
def test_bvm_unnorm_log_prob(batch_dim):
    vm = VonMises(tensor(0.0), tensor(1.0))
    hn = HalfNormal(tensor(1.0))
    b = Beta(tensor(2.0), tensor(2.0))

    while True:
        phi_psi = vm.sample((*batch_dim, 2))
        locs = vm.sample((2, *batch_dim))
        conc = hn.sample((2, *batch_dim))
        corr = b.sample((*batch_dim,))
        if torch.all(torch.prod(conc, dim=0) > corr**2):
            break
    bmv = SineBivariateVonMises(locs[0], locs[1], conc[0], conc[1], corr)
    assert_equal(
        _unnorm_log_prob(phi_psi, locs[0], locs[1], conc[0], conc[1], corr),
        bmv.log_prob(phi_psi) + bmv.norm_const,
    )


def test_bvm_multidim():
    vm = VonMises(tensor(0.0), tensor(1.0))
    hn = HalfNormal(tensor(1.0))
    b = Beta(tensor(2.0), tensor(2.0))
    g = Geometric(torch.tensor([0.4, 0.2, 0.5]))
    for _ in range(25):
        while True:
            batch_dim = tuple(int(i) for i in g.sample() if i > 0)
            sample_dim = tuple(int(i) for i in g.sample() if i > 0)
            locs = vm.sample((2, *batch_dim))
            conc = hn.sample((2, *batch_dim))
            corr = b.sample((*batch_dim,))
            if torch.all(torch.prod(conc, dim=0) > corr**2):
                break

        bmv = SineBivariateVonMises(locs[0], locs[1], conc[0], conc[1], corr)
        assert_equal(bmv.batch_shape, torch.Size(batch_dim))
        assert_equal(
            bmv.sample(sample_dim).shape, torch.Size((*sample_dim, *batch_dim, 2))
        )


def test_mle_bvm():
    vm = VonMises(tensor(0.0), tensor(1.0))
    hn = HalfNormal(tensor(0.8))
    b = Beta(tensor(2.0), tensor(5.0))
    while True:
        locs = vm.sample((2,))
        conc = hn.sample((2,))
        corr = b.sample()
        if torch.prod(conc, dim=-1) >= corr**2:
            break

    def mle_model(data):
        phi_loc = pyro.param("phi_loc", tensor(0.0), constraints.real)
        psi_loc = pyro.param("psi_loc", tensor(0.0), constraints.real)
        phi_conc = pyro.param("phi_conc", tensor(1.0), constraints.positive)
        psi_conc = pyro.param("psi_conc", tensor(1.0), constraints.positive)
        corr = pyro.param("corr", tensor(0.5), constraints.real)
        with pyro.plate("data", data.size(-2)):
            pyro.sample(
                "obs",
                SineBivariateVonMises(phi_loc, psi_loc, phi_conc, psi_conc, corr),
                obs=data,
            )

    def guide(data):
        pass

    bmv = SineBivariateVonMises(locs[0], locs[1], conc[0], conc[1], corr)
    data = bmv.sample((10_000,))

    pyro.clear_param_store()
    adam = pyro.optim.Adam({"lr": 0.01})
    svi = SVI(mle_model, guide, adam, loss=Trace_ELBO())

    losses = []
    steps = 200
    for step in range(steps):
        losses.append(svi.step(data))

    expected = {
        "phi_loc": locs[0],
        "psi_loc": locs[1],
        "phi_conc": conc[0],
        "psi_conc": conc[1],
        "corr": corr,
    }
    actuals = {k: v for k, v in pyro.get_param_store().items()}

    for k in expected.keys():
        if k in actuals:
            actual = actuals[k]
        else:
            actual = (
                actuals["corr_weight"] * actuals["phi_conc"] * actuals["psi_conc"]
            )  # k == 'corr'

        assert_equal(expected[k].squeeze(), actual.squeeze(), 9e-2)


@pytest.mark.parametrize("conc", [1.0, 10.0, 1000.0, 10_000.0, 10_001.0])
def test_sine_bivariate_von_mises_norm(conc):
    if conc > 10_000.0:
        try:
            dist = SineBivariateVonMises(0, 0, conc, conc, 0.0)
            pytest.fail()
        except AssertionError:
            return

    dist = SineBivariateVonMises(0, 0, conc, conc, 0.0)
    num_samples = 500
    x = torch.linspace(-torch.pi, torch.pi, num_samples)
    y = torch.linspace(-torch.pi, torch.pi, num_samples)
    mesh = torch.stack(torch.meshgrid(x, y, indexing="ij"), axis=-1)
    integral_torus = (
        torch.exp(dist.log_prob(mesh)) * (2 * torch.pi) ** 2 / num_samples**2
    ).sum()
    assert torch.allclose(integral_torus, torch.tensor(1.0), rtol=1e-2)
