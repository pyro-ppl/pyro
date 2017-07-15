import torch
import torch.optim
from torch.autograd import Variable
from torch import nn as nn

import pyro
import numpy as np
from pyro.distributions import DiagNormal
from tests.common import TestCase

from pyro.infer.kl_qp import KL_QP
from pyro.util import zeros, ones
from copy import copy


class ParamStoreDictTests(TestCase):

    def setUp(self):
        self.linear_module = nn.Linear(3, 2)
        self.linear_module2 = nn.Linear(3, 2)

    def test_save_and_load(self):
        lin = pyro.module("mymodule", self.linear_module)
        x = Variable(torch.randn(1, 3))
        myparam = pyro.param("myparam", Variable(1.234 * torch.ones(1), requires_grad=True))

        cost = torch.sum(torch.pow(lin(x), 2.0)) * torch.pow(myparam, 4.0)
        cost.backward()
        params = list(self.linear_module.parameters())
        params.append(myparam)
        optim = torch.optim.Adam(params, lr=.01)
        myparam_copy_stale = copy(pyro.param("myparam").data.numpy())

        optim.step()

        myparam_copy = copy(pyro.param("myparam").data.numpy())
        param_store_params = copy(pyro.get_param_store()._params)
        param_store_param_to_name = copy(pyro.get_param_store()._param_to_name)

        pyro.get_param_store().save('paramstore.unittest.out')
        pyro.get_param_store().clear()
        pyro.get_param_store().load('paramstore.unittest.out')

        def modules_are_equal():
            weights_equal = np.sum(np.fabs(self.linear_module2.weight.data.numpy() -
                                   self.linear_module.weight.data.numpy())) == 0.0
            bias_equal = np.sum(np.fabs(self.linear_module2.bias.data.numpy() -
                                self.linear_module.bias.data.numpy())) == 0.0
            return (weights_equal and bias_equal)

        self.assertFalse(modules_are_equal())
        pyro.sync_module("mymodule", self.linear_module2)
        self.assertTrue(modules_are_equal())

        myparam = pyro.param("myparam")
        self.assertFalse(myparam_copy_stale == myparam.data.numpy())
        self.assertTrue(myparam_copy == myparam.data.numpy())
        self.assertTrue(param_store_params.keys().sort() ==
                        pyro.get_param_store()._params.keys().sort())
        self.assertTrue(param_store_param_to_name.values().sort() ==
                        pyro.get_param_store()._param_to_name.values().sort())
