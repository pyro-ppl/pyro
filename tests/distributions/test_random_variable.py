# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch.tensor as tt
import pyro
from pyro.distributions.random_variable import RandomVariable as RV
from pyro.distributions import Uniform

N_SAMPLES = 100


def test_add():
    X = RV(Uniform(0, 1))  # (0, 1)
    X = X + 1  # (1, 2)
    X = 1 + X  # (2, 3)
    X += 1  # (3, 4)
    x = X.sample([N_SAMPLES])
    assert ((3 <= x) & (x <= 4)).all().item()


def test_subtract():
    X = RV(Uniform(0, 1))  # (0, 1)
    X = 1 - X  # (0, 1)
    X = X - 1  # (-1, 0)
    X -= 1  # (-2, -1)
    x = X.sample([N_SAMPLES])
    assert ((-2 <= x) & (x <= -1)).all().item()


def test_multiply_divide():
    X = RV(Uniform(0, 1))  # (0, 1)
    X *= 4  # (0, 4)
    X /= 2  # (0, 2)
    x = X.sample([N_SAMPLES])
    assert ((0 <= x) & (x <= 2)).all().item()


def test_abs():
    X = RV(Uniform(0, 1))  # (0, 1)
    X = 2*(X - 0.5)  # (-1, 1)
    X = abs(X)  # (0, 1)
    x = X.sample([N_SAMPLES])
    assert ((0 <= x) & (x <= 1)).all().item()


def test_neg():
    X = RV(Uniform(0, 1))  # (0, 1)
    X = -X  # (-1, 0)
    x = X.sample([N_SAMPLES])
    assert ((-1 <= x) & (x <= 0)).all().item()


def test_pow():
    X = RV(Uniform(-1, 1))  # (-1, 1)
    X = X**2  # (0, 1)
    x = X.sample([N_SAMPLES])
    assert ((0 <= x) & (x <= 1)).all().item()


def test_tensor_ops():
    pi = 3.141592654
    X = RV(Uniform(0, 1).expand([5, 5]))
    a = tt([[1, 2, 3, 4, 5]])
    b = a.T
    X = abs(pi*(-X + a - 3*b))
    x = X.sample()
    assert x.shape == (5, 5)
    assert (x >= 0).all().item()
