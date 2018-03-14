from __future__ import absolute_import, division, print_function

from collections import namedtuple

import pytest
import torch

from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.likelihoods import Gaussian
from pyro.contrib.gp.models import (GPRegression, SparseGPRegression,
                                    VariationalGP, SparseVariationalGP)
from pyro.optim import Adam
from tests.common import assert_equal

T = namedtuple("TestGPModel", ["model", "X", "y", "kernel", "likelihood"])

X = torch.tensor([[1, 5, 3], [4, 3, 7]])
y1D = torch.tensor([2, 1])
y2D = torch.tensor([[1, 3, 1, -1], [2, 3, 4, 1]])
kernel = RBF(input_dim=3, variance=torch.tensor([1]), lengthscale=torch.tensor([3]))
noise = torch.tensor([1e-6])
likelihood = Gaussian(noise)

TEST_CASES = [
    T(
        GPRegression,
        X, y1D, kernel, noise
    ),
    T(
        GPRegression,
        X, y2D, kernel, noise
    ),
    T(
        SparseGPRegression,
        X, y1D, kernel, noise
    ),
    T(
        SparseGPRegression,
        X, y2D, kernel, noise
    ),
    T(
        VariationalGP,
        X, y1D, kernel, likelihood
    ),
    T(
        VariationalGP,
        X, y2D, kernel, likelihood
    ),
    T(
        SparseVariationalGP,
        X, y1D, kernel, likelihood
    ),
    T(
        SparseVariationalGP,
        X, y2D, kernel, likelihood
    ),
]

TEST_IDS = [t[0].__name__ + "_y{}D".format(str(t[2].dim()))
            for t in TEST_CASES]


@pytest.mark.parametrize("model, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_model_forward(model, X, y, kernel, likelihood):
    if "Sparse" in model.__name__:
        gp = model(X, y, kernel, X, likelihood)
    else:
        gp = model(X, y, kernel, likelihood)

    # test shape
    Xnew = torch.tensor([[2, 3, 1]])
    loc0, cov0 = gp(Xnew, full_cov=True)
    loc1, var1 = gp(Xnew, full_cov=False)
    assert loc0.dim() == y.dim()
    assert loc0.size(0) == Xnew.size(0)
    assert loc0.size()[1:] == y.size()[1:]  # test latent shape
    assert cov0.dim() == 2
    assert cov0.size(0) == cov0.size(1)
    assert cov0.size(0) == Xnew.size(0)
    assert_equal(loc0, loc1)
    assert_equal(cov0.diag(), var1)

    # test trivial forward
    # for variational models, inferences depend on variational parameters, so skip
    if "Variational" in model.__name__:
        pass
    else:
        loc, cov = gp(X, full_cov=True)
        assert_equal(loc, y)
        assert_equal(cov.abs().sum().item(), 0)


@pytest.mark.parametrize("model, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_inference(model, X, y, kernel, likelihood):
    if "Sparse" in model.__name__:
        gp = model(X, y, kernel, X, likelihood)
    else:
        gp = model(X, y, kernel, likelihood)

    gp.optimize(num_steps=1, optimizer=Adam({}))
