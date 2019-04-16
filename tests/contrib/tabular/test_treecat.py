from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.contrib.tabular.features import Boolean, Real
from pyro.contrib.tabular.treecat import TreeCat, TreeCatTrainer
from pyro.distributions.util import default_dtype


with default_dtype(torch.float):
    TINY_DATASETS = [
        [torch.tensor([0., 0., 1.]), torch.tensor([-0.5, 0.5, 10.])],
        [None, torch.tensor([-0.5, 0.5, 10.])],
        [torch.tensor([0., 0., 1.]), None],
    ]


@default_dtype(torch.float)
@pytest.mark.parametrize('data', TINY_DATASETS)
@pytest.mark.parametrize('capacity', [2, 16])
def test_train_smoke(data, capacity):
    features = [Boolean("b"), Real("r")]
    edges = torch.LongTensor([[0, 1]])
    model = TreeCat(features, capacity, edges)
    trainer = TreeCatTrainer(model)
    for i in range(10):
        trainer.step(data)


@default_dtype(torch.float)
@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('data', TINY_DATASETS)
@pytest.mark.parametrize('num_particles', [None, 8])
def test_impute_smoke(data, capacity, num_particles):
    features = [Boolean("b"), Real("r")]
    edges = torch.LongTensor([[0, 1]])
    model = TreeCat(features, capacity, edges)
    model.impute(data, num_particles=num_particles)
