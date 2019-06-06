from __future__ import absolute_import, division, print_function

import os

import pytest
import torch
from six.moves import cPickle as pickle

import pyro
import pyro.poutine as poutine
from pyro.contrib.tabular.features import Boolean, Discrete, Real
from pyro.contrib.tabular.treecat import TreeCat
from pyro.optim import Adam
from tests.common import TemporaryDirectory, assert_close

TINY_DATASETS = [
    {
        "features": [Boolean("f1"), Boolean("f2"), Boolean("f3")],
        "data": [torch.tensor([0., 0., 1.])] * 3,
        "mask": [torch.tensor([True, False, True], dtype=torch.uint8), True, False],
    },
    {
        "features": [Discrete("f1", 3), Discrete("f2", 4), Discrete("f3", 5)],
        "data": [torch.tensor([0, 1, 2, 2, 2])] * 3,
        "mask": [torch.tensor([True, False, True, False, True], dtype=torch.uint8),
                 True, False],
    },
    {
        "features": [Real("f1"), Real("f2"), Real("f3")],
        "data": [torch.tensor([-0.5, 0.5, 10.])] * 3,
        "mask": [torch.tensor([True, False, True], dtype=torch.uint8), True, False],
    },
    {
        "features": [Boolean("f1"), Real("f2"), Discrete("f3", 3), Real("f4"), Boolean("f5")],
        "data": [torch.tensor([0., 0., 0., 1., 1.]),
                 torch.tensor([-1.1, -1.0, -0.9, 0.9, 1.0]),
                 torch.tensor([0, 1, 2, 2, 2], dtype=torch.long),
                 torch.tensor([-2., -1., -0., 1., 2.]),
                 torch.tensor([0., 1., 1., 1., 0.])],
        "mask": [torch.tensor([True, False, True, False, True], dtype=torch.uint8),
                 torch.tensor([True, True, True, False, True], dtype=torch.uint8),
                 torch.tensor([True, False, True, True, True], dtype=torch.uint8),
                 False,
                 True],
    },
]


def train_model(model, data, mask=None):
    trainer = model.trainer(Adam({}))
    trainer.init(data, mask)
    for epoch in range(2):
        trainer.step(data, mask)


@pytest.mark.parametrize('masked', [False, True])
@pytest.mark.parametrize('dataset', TINY_DATASETS)
@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('Model', [TreeCat])
def test_train_smoke(Model, dataset, capacity, masked):
    features = dataset["features"]
    data = dataset["data"]
    mask = dataset["mask"]

    model = Model(features, capacity)
    train_model(model, data, mask if masked else None)


@pytest.mark.parametrize('grad_enabled', [True, False])
@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('dataset', TINY_DATASETS)
@pytest.mark.parametrize('num_samples', [None, 8])
@pytest.mark.parametrize('Model', [TreeCat])
def test_sample_smoke(dataset, Model, capacity, num_samples, grad_enabled):
    features = dataset["features"]
    data = dataset["data"]
    mask = dataset["mask"]
    model = Model(features, capacity)
    train_model(model, data)

    with torch.set_grad_enabled(grad_enabled):
        samples = model.sample(data, mask, num_samples=num_samples)
    assert isinstance(samples, list)
    assert len(samples) == len(features)


@pytest.mark.parametrize('grad_enabled', [True, False])
@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('dataset', TINY_DATASETS)
@pytest.mark.parametrize('Model', [TreeCat])
def test_log_prob_smoke(dataset, Model, capacity, grad_enabled):
    features = dataset["features"]
    data = dataset["data"]
    model = Model(features, capacity)
    train_model(model, data)

    for mask in [None, dataset["mask"]]:
        with torch.set_grad_enabled(grad_enabled):
            loss = model.log_prob(data, mask)
        assert isinstance(loss, torch.Tensor)
        num_rows = len(data[0])
        assert loss.shape == (num_rows,)


@pytest.mark.parametrize('dataset', TINY_DATASETS)
@pytest.mark.parametrize('capacity', [2, 16])
@pytest.mark.parametrize('Model', [TreeCat])
@pytest.mark.parametrize('method', ['pickle', 'torch'])
def test_pickle(method, dataset, Model, capacity):
    features = dataset["features"]
    data = dataset["data"]
    model = Model(features, capacity)
    train_model(model, data)

    if Model is TreeCat:
        expected_edges = model.edges
    expected_trace = poutine.trace(model.guide).get_trace(data)

    with TemporaryDirectory() as path:
        param_filename = os.path.join(path, "model.pyro")
        pickle_filename = os.path.join(path, "model.pkl")
        pyro.get_param_store().save(param_filename)
        with open(pickle_filename, "wb") as f:
            if method == 'pickle':
                pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
            elif method == 'torch':
                torch.save(model, f, pickle_module=pickle,
                           pickle_protocol=pickle.HIGHEST_PROTOCOL)

        pyro.get_param_store().clear()
        del model

        pyro.get_param_store().load(param_filename)
        with open(pickle_filename, "rb") as f:
            if method == 'pickle':
                model = pickle.load(f)
            elif method == 'torch':
                model = torch.load(f, pickle_module=pickle)

    if Model is TreeCat:
        assert (model.edges == expected_edges).all()
    actual_trace = poutine.trace(model.guide).get_trace(data)
    assert expected_trace.nodes.keys() == actual_trace.nodes.keys()
    for key, expected_node in expected_trace.nodes.items():
        if expected_node["type"] in ("param", "sample"):
            actual_node = actual_trace.nodes[key]
            assert_close(expected_node["value"], actual_node["value"])
