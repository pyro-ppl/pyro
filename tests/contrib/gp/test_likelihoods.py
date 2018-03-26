from __future__ import absolute_import, division, print_function

from collections import namedtuple

import pytest
import torch

from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.likelihoods import Binary, MultiClass
from pyro.contrib.gp.models import SparseVariationalGP, VariationalGP

T = namedtuple("TestGPLikelihood", ["model_class", "X", "y", "kernel", "likelihood"])

X = torch.tensor([[1, 5, 3], [4, 3, 7], [3, 4, 6]])
kernel = RBF(input_dim=3, variance=torch.tensor([1]), lengthscale=torch.tensor([3]))
noise = torch.tensor([1e-6])
y_binary = torch.tensor([0, 1, 0])
binary_likelihood = Binary()
y_multiclass = torch.tensor([2, 0, 1])
multiclass_likelihood = MultiClass(num_classes=3)

TEST_CASES = [
    T(
        VariationalGP,
        X, y_binary, kernel, binary_likelihood
    ),
    T(
        VariationalGP,
        X, y_multiclass, kernel, multiclass_likelihood
    ),
    T(
        SparseVariationalGP,
        X, y_binary, kernel, binary_likelihood
    ),
    T(
        SparseVariationalGP,
        X, y_multiclass, kernel, multiclass_likelihood
    ),
]

TEST_IDS = [t[0].__name__ + "_" + t[4].__class__.__name__.split(".")[-1]
            for t in TEST_CASES]


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_inference(model_class, X, y, kernel, likelihood):
    if isinstance(likelihood, MultiClass):
        latent_shape = y.size()[1:] + torch.Size([likelihood.num_classes])
    else:
        latent_shape = y.size()[1:]
    if model_class is SparseVariationalGP:
        gp = model_class(X, y, kernel, X, likelihood, latent_shape)
    else:
        gp = model_class(X, y, kernel, likelihood, latent_shape)

    gp.optimize(num_steps=1)
