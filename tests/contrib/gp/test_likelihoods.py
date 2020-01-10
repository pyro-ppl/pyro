# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple

import pytest
import torch

from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.likelihoods import Binary, MultiClass, Poisson
from pyro.contrib.gp.models import VariationalGP, VariationalSparseGP
from pyro.contrib.gp.util import train


T = namedtuple("TestGPLikelihood", ["model_class", "X", "y", "kernel", "likelihood"])

X = torch.tensor([[1.0, 5.0, 3.0], [4.0, 3.0, 7.0], [3.0, 4.0, 6.0]])
kernel = RBF(input_dim=3, variance=torch.tensor(1.), lengthscale=torch.tensor(3.))
noise = torch.tensor(1e-6)
y_binary1D = torch.tensor([0.0, 1.0, 0.0])
y_binary2D = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
binary_likelihood = Binary()
y_count1D = torch.tensor([0.0, 1.0, 4.0])
y_count2D = torch.tensor([[5.0, 9.0, 3.0], [4.0, 0.0, 1.0]])
poisson_likelihood = Poisson()
y_multiclass1D = torch.tensor([2.0, 0.0, 1.0])
y_multiclass2D = torch.tensor([[2.0, 1.0, 1.0], [0.0, 2.0, 1.0]])
multiclass_likelihood = MultiClass(num_classes=3)

TEST_CASES = [
    T(
        VariationalGP,
        X, y_binary1D, kernel, binary_likelihood
    ),
    T(
        VariationalGP,
        X, y_binary2D, kernel, binary_likelihood
    ),
    T(
        VariationalGP,
        X, y_multiclass1D, kernel, multiclass_likelihood
    ),
    T(
        VariationalGP,
        X, y_multiclass2D, kernel, multiclass_likelihood
    ),
    T(
        VariationalGP,
        X, y_count1D, kernel, poisson_likelihood
    ),
    T(
        VariationalGP,
        X, y_count2D, kernel, poisson_likelihood
    ),
    T(
        VariationalSparseGP,
        X, y_binary1D, kernel, binary_likelihood
    ),
    T(
        VariationalSparseGP,
        X, y_binary2D, kernel, binary_likelihood
    ),
    T(
        VariationalSparseGP,
        X, y_multiclass1D, kernel, multiclass_likelihood
    ),
    T(
        VariationalSparseGP,
        X, y_multiclass2D, kernel, multiclass_likelihood
    ),
    T(
        VariationalSparseGP,
        X, y_count1D, kernel, poisson_likelihood
    ),
    T(
        VariationalSparseGP,
        X, y_count2D, kernel, poisson_likelihood
    ),
]

TEST_IDS = ["_".join([t[0].__name__, t[4].__class__.__name__.split(".")[-1],
                      str(t[2].dim()) + "D"])
            for t in TEST_CASES]


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_inference(model_class, X, y, kernel, likelihood):
    if isinstance(likelihood, MultiClass):
        latent_shape = y.shape[:-1] + (likelihood.num_classes,)
    else:
        latent_shape = y.shape[:-1]
    if model_class is VariationalSparseGP:
        gp = model_class(X, y, kernel, X, likelihood, latent_shape=latent_shape)
    else:
        gp = model_class(X, y, kernel, likelihood, latent_shape=latent_shape)

    train(gp, num_steps=1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_inference_with_empty_latent_shape(model_class, X, y, kernel, likelihood):
    if isinstance(likelihood, MultiClass):
        latent_shape = torch.Size([likelihood.num_classes])
    else:
        latent_shape = torch.Size([])
    if model_class is VariationalSparseGP:
        gp = model_class(X, y, kernel, X, likelihood, latent_shape=latent_shape)
    else:
        gp = model_class(X, y, kernel, likelihood, latent_shape=latent_shape)

    train(gp, num_steps=1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_forward(model_class, X, y, kernel, likelihood):
    if isinstance(likelihood, MultiClass):
        latent_shape = y.shape[:-1] + (likelihood.num_classes,)
    else:
        latent_shape = y.shape[:-1]
    if model_class is VariationalSparseGP:
        gp = model_class(X, y, kernel, X, likelihood, latent_shape=latent_shape)
    else:
        gp = model_class(X, y, kernel, likelihood, latent_shape=latent_shape)

    Xnew_shape = (X.shape[0] * 2,) + X.shape[1:]
    Xnew = torch.rand(Xnew_shape, dtype=X.dtype, device=X.device)
    f_loc, f_var = gp(Xnew)
    ynew = gp.likelihood(f_loc, f_var)

    assert ynew.shape == y.shape[:-1] + (Xnew.shape[0],)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_forward_with_empty_latent_shape(model_class, X, y, kernel, likelihood):
    if isinstance(likelihood, MultiClass):
        latent_shape = torch.Size([likelihood.num_classes])
    else:
        latent_shape = torch.Size([])
    if model_class is VariationalSparseGP:
        gp = model_class(X, y, kernel, X, likelihood, latent_shape=latent_shape)
    else:
        gp = model_class(X, y, kernel, likelihood, latent_shape=latent_shape)

    Xnew_shape = (X.shape[0] * 2,) + X.shape[1:]
    Xnew = torch.rand(Xnew_shape, dtype=X.dtype, device=X.device)
    f_loc, f_var = gp(Xnew)
    ynew = gp.likelihood(f_loc, f_var)

    assert ynew.shape == (Xnew.shape[0],)
