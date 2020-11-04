from contextlib import ExitStack

import pytest
import torch
import torch.tensor as tt

import pyro
from pyro.distributions import Normal, Exponential
from pyro.infer import log_likelihood


def model1(x):
    m = pyro.sample("m", Normal(0., 1.))
    b = pyro.sample("b", Normal(0., 1.))
    sigma = pyro.sample("sigma", Exponential(1.))
    # gotta do some tricky stuff to support dynamic batch sizes...
    plates = reversed([pyro.plate(f"obs_{i}", d) for i, d in enumerate(x.shape)])
    with ExitStack() as stack:
        for cm in plates:
            stack.enter_context(cm)
        y = m*x + b
        pyro.sample("y", Normal(y, sigma))


def guide1(x):
    m_mean = pyro.param("m_mean", tt(0.))
    m_std = pyro.param("m_std", tt(1.))
    b_mean = pyro.param("b_mean", tt(0.))
    b_std = pyro.param("b_std", tt(1.))
    sigma_mean = pyro.param("sigma_mean", tt(1.))
    # draw samples!
    pyro.sample("m", Normal(m_mean, m_std))
    pyro.sample("b", Normal(b_mean, b_std))
    pyro.sample("sigma", Exponential(sigma_mean))


@pytest.mark.parametrize("batch_shape", [(), (1,), (5,), (10,), (3, 4), (1, 2, 3, 4)])
@pytest.mark.parametrize("num_post_samples", [1, 5, 10])
@pytest.mark.parametrize("parallel", [True, False])
def test_batch_shape(batch_shape, num_post_samples, parallel):
    x = torch.randn(batch_shape)
    y = x + 0.1 * torch.randn(batch_shape)

    cond_model = pyro.condition(model1, data={"y": y})

    lglik = log_likelihood(
        cond_model,
        guide=guide1,
        num_samples=num_post_samples,
        parallel=parallel,
    )(x)

    assert len(lglik) == 1
    assert "y" in lglik
    assert lglik["y"].shape == (num_post_samples,) + batch_shape


def model2(*Xs):
    m = pyro.sample("m", Normal(0., 1.))
    b = pyro.sample("b", Normal(0., 1.))
    sigma = pyro.sample("sigma", Exponential(1.))
    with pyro.plate("obs", len(Xs[0])):
        for i, x in enumerate(Xs):
            y = m * x + b
            pyro.sample(f"y_{i}", Normal(y, sigma))


def guide2(*Xs):
    m_mean = pyro.param("m_mean", tt(0.))
    m_std = pyro.param("m_std", tt(1.))
    b_mean = pyro.param("b_mean", tt(0.))
    b_std = pyro.param("b_std", tt(1.))
    sigma_mean = pyro.param("sigma_mean", tt(1.))
    # draw samples!
    pyro.sample("m", Normal(m_mean, m_std))
    pyro.sample("b", Normal(b_mean, b_std))
    pyro.sample("sigma", Exponential(sigma_mean))


@pytest.mark.parametrize("num_sites", [1, 5, 10])
@pytest.mark.parametrize("num_post_samples", [1, 7, 15])
@pytest.mark.parametrize("parallel", [True, False])
def test_obs_sites(num_sites, num_post_samples, parallel):
    N = 20
    x = torch.randn((num_sites, N))
    y = x + 0.1 * torch.randn((num_sites, N))

    Xs = [x[i] for i in range(num_sites)]
    data = {f"y_{i}": y[i] for i in range(num_sites)}
    cond_model = pyro.condition(model2, data=data)

    lglik = log_likelihood(
        cond_model,
        guide=guide2,
        num_samples=num_post_samples,
        parallel=parallel,
    )(*Xs)

    assert len(lglik) == num_sites
    assert set(lglik.keys()) == set(data.keys())
    assert all(lglik[k].shape == (num_post_samples, N) for k in lglik)
