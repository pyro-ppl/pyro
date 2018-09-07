from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.contrib.tabular import Boolean, Real, TreeCat
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam

TINY_DATASETS = [
    [torch.tensor([0., 0., 1.]), torch.tensor([-0.5, 0.5, 10.])],
    [None, torch.tensor([-0.5, 0.5, 10.])],
    [torch.tensor([0., 0., 1.]), None],
]


@pytest.mark.parametrize('data', TINY_DATASETS)
@pytest.mark.parametrize('capacity', [2, 16])
def test_svi_smoke(data, capacity):
    features = [Boolean("b"), Real("r")]
    edges = ((0, 1),)
    treecat = TreeCat(features, capacity, edges)
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    optim = Adam({'lr': 0.01})
    svi = SVI(treecat.model, treecat.guide, optim, elbo)
    for i in range(10):
        svi.step(data, impute=False)


@pytest.mark.parametrize('data', TINY_DATASETS)
@pytest.mark.parametrize('capacity', [2, 16])
def test_predict_smoke(data, capacity):
    features = [Boolean("b"), Real("r")]
    edges = ((0, 1),)
    treecat = TreeCat(features, capacity, edges)
    elbo = TraceEnum_ELBO(max_iarange_nesting=1)
    elbo.sample_posterior(treecat.model, treecat.guide, data, impute=True)
