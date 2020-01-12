# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
import pytest
import torch

from pyro.contrib.util import (
    get_indices, tensor_to_dict, rmv, rvv, lexpand, rexpand, rdiag, rtril
)
from tests.common import assert_equal


def test_get_indices_sizes():
    sizes = OrderedDict([("a", 2), ("b", 2), ("c", 2)])
    assert_equal(get_indices(["b"], sizes=sizes), torch.tensor([2, 3]))
    assert_equal(get_indices(["b", "c"], sizes=sizes), torch.tensor([2, 3, 4, 5]))
    tensors = OrderedDict([("a", torch.ones(2)), ("b", torch.ones(2)), ("c", torch.ones(2))])
    assert_equal(get_indices(["b"], tensors=tensors), torch.tensor([2, 3]))
    assert_equal(get_indices(["b", "c"], tensors=tensors), torch.tensor([2, 3, 4, 5]))


def test_tensor_to_dict():
    sizes = OrderedDict([("a", 2), ("b", 2), ("c", 2)])
    vector = torch.tensor([1., 2, 3, 4, 5, 6])
    assert_equal(tensor_to_dict(sizes, vector), {"a": torch.tensor([1., 2.]),
                                                 "b": torch.tensor([3., 4.]),
                                                 "c": torch.tensor([5., 6.])})
    assert_equal(tensor_to_dict(sizes, vector, subset=["b"]),
                 {"b": torch.tensor([3., 4.])})


@pytest.mark.parametrize("A,b", [
    (torch.tensor([[1., 2.], [2., -3.]]), torch.tensor([-1., 2.]))
    ])
def test_rmv(A, b):
    assert_equal(rmv(A, b), A.mv(b), prec=1e-8)
    batched_A = lexpand(A, 5, 4)
    batched_b = lexpand(b, 5, 4)
    expected_Ab = lexpand(A.mv(b), 5, 4)
    assert_equal(rmv(batched_A, batched_b), expected_Ab, prec=1e-8)


@pytest.mark.parametrize("a,b", [
    (torch.tensor([1., 2.]), torch.tensor([-1., 2.]))
    ])
def test_rvv(a, b):
    assert_equal(rvv(a, b), torch.dot(a, b), prec=1e-8)
    batched_a = lexpand(a, 5, 4)
    batched_b = lexpand(b, 5, 4)
    expected_ab = lexpand(torch.dot(a, b), 5, 4)
    assert_equal(rvv(batched_a, batched_b), expected_ab, prec=1e-8)


def test_lexpand():
    A = torch.tensor([[1., 2.], [-2., 0]])
    assert_equal(lexpand(A), A, prec=1e-8)
    assert_equal(lexpand(A, 4), A.expand(4, 2, 2), prec=1e-8)
    assert_equal(lexpand(A, 4, 2), A.expand(4, 2, 2, 2), prec=1e-8)


def test_rexpand():
    A = torch.tensor([[1., 2.], [-2., 0]])
    assert_equal(rexpand(A), A, prec=1e-8)
    assert_equal(rexpand(A, 4), A.unsqueeze(-1).expand(2, 2, 4), prec=1e-8)
    assert_equal(rexpand(A, 4, 2), A.unsqueeze(-1).unsqueeze(-1).expand(2, 2, 4, 2), prec=1e-8)


def test_rtril():
    A = torch.tensor([[1., 2.], [-2., 0]])
    assert_equal(rtril(A), torch.tril(A), prec=1e-8)
    expanded = lexpand(A, 5, 4)
    expected = lexpand(torch.tril(A), 5, 4)
    assert_equal(rtril(expanded), expected, prec=1e-8)


def test_rdiag():
    v = torch.tensor([1., 2., -1.])
    assert_equal(rdiag(v), torch.diag(v), prec=1e-8)
    expanded = lexpand(v, 5, 4)
    expeceted = lexpand(torch.diag(v), 5, 4)
    assert_equal(rdiag(expanded), expeceted, prec=1e-8)
