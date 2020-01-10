# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from unittest import TestCase

import numpy as np
import torch
import torch.optim
from torch import nn as nn
from torch.distributions import constraints

import pyro
from tests.common import assert_equal


class ParamStoreDictTests(TestCase):

    def setUp(self):
        pyro.clear_param_store()
        self.linear_module = nn.Linear(3, 2)
        self.linear_module2 = nn.Linear(3, 2)
        self.linear_module3 = nn.Linear(3, 2)

    def test_save_and_load(self):
        lin = pyro.module("mymodule", self.linear_module)
        pyro.module("mymodule2", self.linear_module2)
        x = torch.randn(1, 3)
        myparam = pyro.param("myparam", 1.234 * torch.ones(1))

        cost = torch.sum(torch.pow(lin(x), 2.0)) * torch.pow(myparam, 4.0)
        cost.backward()
        params = list(self.linear_module.parameters()) + [myparam]
        optim = torch.optim.Adam(params, lr=.01)
        myparam_copy_stale = copy(pyro.param("myparam").detach().cpu().numpy())

        optim.step()

        myparam_copy = copy(pyro.param("myparam").detach().cpu().numpy())
        param_store_params = copy(pyro.get_param_store()._params)
        param_store_param_to_name = copy(pyro.get_param_store()._param_to_name)
        assert len(list(param_store_params.keys())) == 5
        assert len(list(param_store_param_to_name.values())) == 5

        pyro.get_param_store().save('paramstore.unittest.out')
        pyro.clear_param_store()
        assert len(list(pyro.get_param_store()._params)) == 0
        assert len(list(pyro.get_param_store()._param_to_name)) == 0
        pyro.get_param_store().load('paramstore.unittest.out')

        def modules_are_equal():
            weights_equal = np.sum(np.fabs(self.linear_module3.weight.detach().cpu().numpy() -
                                   self.linear_module.weight.detach().cpu().numpy())) == 0.0
            bias_equal = np.sum(np.fabs(self.linear_module3.bias.detach().cpu().numpy() -
                                self.linear_module.bias.detach().cpu().numpy())) == 0.0
            return (weights_equal and bias_equal)

        assert not modules_are_equal()
        pyro.module("mymodule", self.linear_module3, update_module_params=False)
        assert id(self.linear_module3.weight) != id(pyro.param('mymodule$$$weight'))
        assert not modules_are_equal()
        pyro.module("mymodule", self.linear_module3, update_module_params=True)
        assert id(self.linear_module3.weight) == id(pyro.param('mymodule$$$weight'))
        assert modules_are_equal()

        myparam = pyro.param("myparam")
        store = pyro.get_param_store()
        assert myparam_copy_stale != myparam.detach().cpu().numpy()
        assert myparam_copy == myparam.detach().cpu().numpy()
        assert sorted(param_store_params.keys()) == sorted(store._params.keys())
        assert sorted(param_store_param_to_name.values()) == sorted(store._param_to_name.values())
        assert sorted(store._params.keys()) == sorted(store._param_to_name.values())


def test_dict_interface():
    param_store = pyro.get_param_store()

    # start empty
    param_store.clear()
    assert not param_store
    assert len(param_store) == 0
    assert 'x' not in param_store
    assert 'y' not in param_store
    assert list(param_store.items()) == []
    assert list(param_store.keys()) == []
    assert list(param_store.values()) == []

    # add x
    param_store['x'] = torch.zeros(1, 2, 3)
    assert param_store
    assert len(param_store) == 1
    assert 'x' in param_store
    assert 'y' not in param_store
    assert list(param_store.keys()) == ['x']
    assert [key for key, value in param_store.items()] == ['x']
    assert len(list(param_store.values())) == 1
    assert param_store['x'].shape == (1, 2, 3)
    assert_equal(param_store.setdefault('x', torch.ones(1, 2, 3)), torch.zeros(1, 2, 3))
    assert param_store['x'].unconstrained() is param_store['x']

    # add y
    param_store.setdefault('y', torch.ones(4, 5), constraint=constraints.positive)
    assert param_store
    assert len(param_store) == 2
    assert 'x' in param_store
    assert 'y' in param_store
    assert sorted(param_store.keys()) == ['x', 'y']
    assert sorted(key for key, value in param_store.items()) == ['x', 'y']
    assert len(list(param_store.values())) == 2
    assert param_store['x'].shape == (1, 2, 3)
    assert param_store['y'].shape == (4, 5)
    assert_equal(param_store.setdefault('y', torch.zeros(4, 5)), torch.ones(4, 5))
    assert_equal(param_store['y'].unconstrained(), torch.zeros(4, 5))

    # remove x
    del param_store['x']
    assert param_store
    assert len(param_store) == 1
    assert 'x' not in param_store
    assert 'y' in param_store
    assert list(param_store.keys()) == ['y']
    assert list(key for key, value in param_store.items()) == ['y']
    assert len(list(param_store.values())) == 1
    assert param_store['y'].shape == (4, 5)
    assert_equal(param_store.setdefault('y', torch.zeros(4, 5)), torch.ones(4, 5))
    assert_equal(param_store['y'].unconstrained(), torch.zeros(4, 5))

    # remove y
    del param_store['y']
    assert not param_store
    assert len(param_store) == 0
    assert 'x' not in param_store
    assert 'y' not in param_store
    assert list(param_store.keys()) == []
    assert list(key for key, value in param_store.items()) == []
    assert len(list(param_store.values())) == 0
