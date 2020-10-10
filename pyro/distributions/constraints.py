# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions.constraints import *  # noqa F403
from torch.distributions.constraints import Constraint
from torch.distributions.constraints import __all__ as torch_constraints
from torch.distributions.constraints import lower_cholesky


# TODO move this upstream to torch.distributions
class IndependentConstraint(Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.

    :param torch.distributions.constraints.Constraint base_constraint: A base
        constraint whose entries are incidentally independent.
    :param int reinterpreted_batch_ndims: The number of extra event dimensions that will
        be considered dependent.
    """
    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def check(self, value):
        result = self.base_constraint.check(value)
        result = result.reshape(result.shape[:result.dim() - self.reinterpreted_batch_ndims] + (-1,))
        result = result.min(-1)[0]
        return result


# TODO move this upstream to torch.distributions
class _Integer(Constraint):
    """
    Constrain to integers.
    """
    def check(self, value):
        return value % 1 == 0

    def __repr__(self):
        return self.__class__.__name__[1:]


class _CorrCholesky(Constraint):
    """
    Constrains to lower-triangular square matrices with positive diagonals and
    Euclidean norm of each row is 1, such that `torch.mm(omega, omega.t())` will
    have unit diagonal.
    """

    def check(self, value):
        unit_norm_row = (value.norm(dim=-1).sub(1) < 1e-4).min(-1)[0]
        return lower_cholesky.check(value) & unit_norm_row


class _OrderedVector(Constraint):
    """
    Constrains to a real-valued tensor where the elements are monotonically
    increasing along the `event_shape` dimension.
    """

    def check(self, value):
        if value.ndim == 0:
            return torch.tensor(False, device=value.device)
        elif value.shape[-1] == 1:
            return torch.ones_like(value[..., 0], dtype=bool)
        else:
            return torch.all(value[..., 1:] > value[..., :-1], dim=-1)


corr_cholesky_constraint = _CorrCholesky()
integer = _Integer()
ordered_vector = _OrderedVector()

__all__ = [
    'IndependentConstraint',
    'corr_cholesky_constraint',
    'integer',
    'ordered_vector',
]

__all__.extend(torch_constraints)
__all__ = sorted(set(__all__))
del torch_constraints


# Create sphinx documentation.
__doc__ = """
    Pyro's constraints library extends
    :mod:`torch.distributions.constraints`.
"""
__doc__ += "\n".join([
    """
    {}
    ----------------------------------------------------------------
    {}
    """.format(
        _name,
        "alias of :class:`torch.distributions.constraints.{}`".format(_name)
        if globals()[_name].__module__.startswith("torch") else
        ".. autoclass:: {}".format(_name)
    )
    for _name in sorted(__all__)
])
