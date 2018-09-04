from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import pytest
import torch

from pyro.infer.contract import _partition_terms, contract_to_tensor
from pyro.poutine.indep_messenger import CondIndepStackFrame


@pytest.mark.parametrize('shapes,dims,expected_num_components', [
    ([()], set(), 1),
    ([(2,)], set(), 1),
    ([(2,)], set([-1]), 1),
    ([(2,), (2,)], set(), 2),
    ([(2,), (2,)], set([-1]), 1),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set(), 4),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set([-1]), 3),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set([-2]), 3),
    ([(2, 1), (2, 1), (1, 3), (1, 3)], set([-1, -2]), 2),
    ([(2, 1), (2, 3), (1, 3)], set(), 3),
    ([(2, 1), (2, 3), (1, 3)], set([-1]), 2),
    ([(2, 1), (2, 3), (1, 3)], set([-2]), 2),
    ([(2, 1), (2, 3), (1, 3)], set([-1, -2]), 1),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set(), 4),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-1]), 3),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-2]), 3),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-3]), 3),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-1, -3]), 2),
    ([(4, 1, 1), (4, 3, 1), (1, 3, 2), (1, 1, 2)], set([-1, -2, -3]), 1),
])
def test_partition_terms(shapes, dims, expected_num_components):
    tensors = [torch.randn(shape) for shape in shapes]
    components = list(_partition_terms(tensors, dims))

    # Check that result is a partition.
    expected_terms = sorted(tensors, key=id)
    actual_terms = sorted((x for c in components for x in c[0]), key=id)
    assert actual_terms == expected_terms
    assert dims == set.union(set(), *(c[1] for c in components))

    # Check that partition is not too coarse.
    assert len(components) == expected_num_components

    # Check that partition is not too fine.
    component_dict = {x: i for i, (terms, _) in enumerate(components) for x in terms}
    for x in tensors:
        for y in tensors:
            if x is not y:
                if any(dim >= -x.dim() and x.shape[dim] > 1 and
                       dim >= -y.dim() and y.shape[dim] > 1 for dim in dims):
                    assert component_dict[x] == component_dict[y]


def frame(dim, size):
    return CondIndepStackFrame(name="iarange_{}".format(size), dim=dim, size=size, counter=0)


EXAMPLES = [
    {
        'shape_tree': {
            (): [(2, 3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': [],
        'target_ordinal': (),
        'expected_shape': (2, 3, 1),
    },
    {
        'shape_tree': {
            (): [(2, 3, 1)],
            (frame(-1, 4),): [(2, 3, 4)],
        },
        'sum_dims': [],
        'target_ordinal': (frame(-1, 4),),
        'expected_shape': (2, 3, 4),
    },
    # ------------------------------------------------------
    #          z
    #          | 4    max_iarange_nesting=2
    #    x     y      w, x, y, z are all enumerated in dims:
    #   2 \   / 3    -3 -4 -5 -6
    #       w
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for w
        'sum_dims': [-4, -5, -6],
        'target_ordinal': (),
        'expected_shape': (2, 1, 1),
    },
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for x
        'sum_dims': [-3, -5, -6],
        'target_ordinal': (frame(-1, 2),),
        'expected_shape': (2, 1, 1, 2),
    },
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for x
        'sum_dims': [-3, -4, -6],
        'target_ordinal': (frame(-1, 3),),
        'expected_shape': (2, 1, 1, 1, 3),
    },
    {
        'shape_tree': {
            (): [(2, 1, 1)],  # w
            (frame(-1, 2),): [(2, 2, 1, 2)],  # x
            (frame(-1, 3),): [(2, 1, 2, 1, 3)],  # y
            (frame(-1, 3), frame(-2, 4)): [(2, 2, 1, 1, 4, 3)],  # z
        },
        # query for z
        'sum_dims': [-3, -4, -5],
        'target_ordinal': (frame(-1, 3), frame(-2, 4)),
        'expected_shape': (2, 1, 1, 1, 4, 3),
    },
]


@pytest.mark.parametrize('example', EXAMPLES)
def test_contract_to_tensor(example):
    tensor_tree = OrderedDict((frozenset(t), [torch.randn(shape) for shape in shapes])
                              for t, shapes in example['shape_tree'].items())
    sum_dims = {x: set(d for d in example['sum_dims'] if -d <= x.dim() and x.shape[d] > 1)
                for terms in tensor_tree.values()
                for x in terms}
    target_ordinal = frozenset(example['target_ordinal'])
    expected_shape = example['expected_shape']

    actual = contract_to_tensor(tensor_tree, sum_dims, target_ordinal)
    assert actual.shape == expected_shape
