from __future__ import absolute_import, division, print_function

import os

import pytest
import torch
from six.moves import cPickle as pickle

import pyro
import pyro.poutine as poutine
from pyro.contrib.tabular.features import Boolean, Discrete, Real
from pyro.contrib.tabular.mixture import Mixture
from pyro.contrib.tabular.treecat import TreeCat
from tests.common import TemporaryDirectory, assert_close

TINY_SCHEMA = [Boolean("f1"), Real("f2"), Discrete("f3", 3), Real("f4"), Boolean("f5")]
TINY_DATASETS = [
    [torch.tensor([0., 0., 1.]), None],
    [None, torch.tensor([-0.5, 0.5, 10.])],
    [None, None, torch.tensor([0, 1, 2, 2, 2], dtype=torch.long)],
    [torch.tensor([0., 0., 1.]), torch.tensor([-0.5, 0.5, 10.])],
    [torch.tensor([0., 0., 0., 1., 1.]),
     torch.tensor([-1.1, -1.0, -0.9, 0.9, 1.0]),
     torch.tensor([0, 1, 2, 2, 2], dtype=torch.long),
     torch.tensor([-2., -1., -0., 1., 2.]),
     torch.tensor([0., 1., 1., 1., 0.])],
]


@pytest.mark.parametrize('data', TINY_DATASETS)
@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('Model', [Mixture, TreeCat])
def test_train_smoke(Model, data, capacity):
    V = len(data)
    features = TINY_SCHEMA[:V]
    model = Model(features, capacity)
    trainer = model.trainer()
    trainer.init(data)
    for i in range(10):
        trainer.step(data)


@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('data', TINY_DATASETS)
@pytest.mark.parametrize('num_samples', [None, 8])
@pytest.mark.parametrize('Model', [Mixture, TreeCat])
def test_impute_smoke(data, Model, capacity, num_samples):
    features = TINY_SCHEMA[:len(data)]
    model = Model(features, capacity)
    model.impute(data, num_samples=num_samples)


@pytest.mark.parametrize('data', TINY_DATASETS)
@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('Model', [Mixture, TreeCat])
def test_pickle(data, Model, capacity):

    def train_model():
        V = len(data)
        features = TINY_SCHEMA[:V]
        model = Model(features, capacity)
        trainer = model.trainer()
        trainer.init(data)
        trainer.step(data)
        return model

    model = train_model()
    if Model is TreeCat:
        expected_edges = model.edges
    expected_trace = poutine.trace(model.guide).get_trace(data)

    with TemporaryDirectory() as path:
        param_filename = os.path.join(path, "model.pyro")
        pickle_filename = os.path.join(path, "model.pkl")
        pyro.get_param_store().save(param_filename)
        with open(pickle_filename, "wb") as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

        pyro.get_param_store().clear()
        del model

        pyro.get_param_store().load(param_filename)
        with open(pickle_filename, "rb") as f:
            model = pickle.load(f)

    if Model is TreeCat:
        assert (model.edges == expected_edges).all()
    actual_trace = poutine.trace(model.guide).get_trace(data)
    assert expected_trace.nodes.keys() == actual_trace.nodes.keys()
    for key, expected_node in expected_trace.nodes.items():
        if expected_node["type"] in ("param", "sample"):
            actual_node = actual_trace.nodes[key]
            assert_close(expected_node["value"], actual_node["value"])
