# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
import pyro.optim


class outest(nn.Module):

    def __init__(self):
        super().__init__()
        self.l0 = outer()
        self.l1 = nn.Linear(2, 2)
        self.l2 = inner()

    def forward(self, s):
        pass


class outer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l0 = inner()
        self.l1 = nn.Linear(2, 2)

    def forward(self, s):
        pass


class inner(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(2, 2)
        self.l1 = nn.ReLU()

    def forward(self, s):
        pass


sequential = nn.Sequential(
          nn.Conv2d(1, 20, 5),
          nn.ReLU(),
          nn.Conv2d(20, 64, 5)
          )


@pytest.mark.parametrize("nn_module", [outest, outer])
def test_module_nn(nn_module):
    pyro.clear_param_store()
    nn_module = nn_module()
    assert pyro.get_param_store()._params == {}
    pyro.module("module", nn_module)
    for name in pyro.get_param_store():
        assert pyro.params.user_param_name(name) in nn_module.state_dict().keys()


@pytest.mark.parametrize("nn_module", [outest, outer])
def test_param_no_grad(nn_module):
    class net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = Parameter(torch.zeros(1))
            self.y = Parameter(torch.zeros(1), requires_grad=False)

        def forward(self, s):
            pass

    with pytest.warns(UserWarning):
        pyro.module('net', net())
    assert 'net$$$x' in pyro.get_param_store().keys()
    assert 'net$$$y' not in pyro.get_param_store().keys()


@pytest.mark.parametrize("nn_module", [sequential])
def test_module_sequential(nn_module):
    pyro.clear_param_store()
    assert pyro.get_param_store()._params == {}
    pyro.module("module", nn_module)
    for name in pyro.get_param_store():
        assert pyro.params.user_param_name(name) in nn_module.state_dict().keys()


@pytest.mark.parametrize("nn_module", [outest, outer])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_random_module(nn_module):
    pyro.clear_param_store()
    nn_module = nn_module()
    p = torch.ones(2, 2)
    prior = dist.Bernoulli(p)
    lifted_mod = pyro.random_module("module", nn_module, prior)
    nn_module = lifted_mod()
    for name, parameter in nn_module.named_parameters():
        assert torch.equal(torch.ones(2, 2), parameter.data)
