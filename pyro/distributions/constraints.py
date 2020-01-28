# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

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


class _CorrCholesky(Constraint):
    """
    Constrains to lower-triangular square matrices with positive diagonals and
    Euclidean norm of each row is 1, such that `torch.mm(omega, omega.t())` will
    have unit diagonal.
    """

    def check(self, value):
        unit_norm_row = (value.norm(dim=-1).sub(1) < 1e-4).min(-1)[0]
        return lower_cholesky.check(value) & unit_norm_row


corr_cholesky_constraint = _CorrCholesky()

__all__ = [
    'IndependentConstraint',
    'corr_cholesky_constraint',
]

__all__.extend(torch_constraints)
del torch_constraints
