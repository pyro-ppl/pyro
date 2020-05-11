# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
import logging

import torch

import funsor
from funsor.domains import bint, reals
from funsor.tensor import Tensor

import pyro.contrib.funsor
from pyro.contrib.funsor.handlers.named_messenger import NamedMessenger

from pyroapi import pyro, pyro_backend


funsor.set_backend("torch")

logger = logging.getLogger(__name__)


def test_iteration():

    def testing():
        for i in pyro.markov(range(5)):
            v1 = pyro.to_data(Tensor(torch.ones(2), OrderedDict([(str(i), bint(2))]), 'real'))
            v2 = pyro.to_data(Tensor(torch.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
            fv1 = pyro.to_funsor(v1, reals())
            fv2 = pyro.to_funsor(v2, reals())
            print(i, v1.shape)  # shapes should alternate
            if i % 2 == 0:
                assert v1.shape == (2,)
            else:
                assert v1.shape == (2, 1, 1)
            assert v2.shape == (2, 1)
            print(i, fv1.inputs)
            print('a', v2.shape)  # shapes should stay the same
            print('a', fv2.inputs)

    with pyro_backend("contrib.funsor"), NamedMessenger():
        testing()


def test_nesting():

    def testing():

        with pyro.markov():
            v1 = pyro.to_data(Tensor(torch.ones(2), OrderedDict([("1", bint(2))]), 'real'))
            print(1, v1.shape)  # shapes should alternate
            assert v1.shape == (2,)

            with pyro.markov():
                v2 = pyro.to_data(Tensor(torch.ones(2), OrderedDict([("2", bint(2))]), 'real'))
                print(2, v2.shape)  # shapes should alternate
                assert v2.shape == (2, 1)

                with pyro.markov():
                    v3 = pyro.to_data(Tensor(torch.ones(2), OrderedDict([("3", bint(2))]), 'real'))
                    print(3, v3.shape)  # shapes should alternate
                    assert v3.shape == (2,)

                    with pyro.markov():
                        v4 = pyro.to_data(Tensor(torch.ones(2), OrderedDict([("4", bint(2))]), 'real'))
                        print(4, v4.shape)  # shapes should alternate

                        assert v4.shape == (2, 1)

    with pyro_backend("contrib.funsor"), NamedMessenger():
        testing()


def test_staggered():

    def testing():
        for i in pyro.markov(range(12)):
            if i % 4 == 0:
                v2 = pyro.to_data(Tensor(torch.zeros(2), OrderedDict([('a', bint(2))]), 'real'))
                fv2 = pyro.to_funsor(v2, reals())
                assert v2.shape == (2,)
                print('a', v2.shape)
                print('a', fv2.inputs)

    with pyro_backend("contrib.funsor"), NamedMessenger():
        testing()
