# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints

from pyro.distributions.constraints import corr_cholesky_constraint
from pyro.distributions.torch import Beta
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.transforms.cholesky import _vector_to_l_cholesky


# TODO: Modify class to support more than one eta value at a time?
class LKJCorrCholesky(TorchDistribution):
    """
    Generates cholesky factors of correlation matrices using an LKJ prior.

    The expected use is to combine it with a vector of variances and pass it
    to the scale_tril parameter of a multivariate distribution such as MultivariateNormal.

    E.g., if theta is a (positive) vector of covariances with the same dimensionality
    as this distribution, and Omega is sampled from this distribution,
    scale_tril=torch.mm(torch.diag(sqrt(theta)), Omega)

    Note that the `event_shape` of this distribution is `[d, d]`

    .. note::

       When using this distribution with HMC/NUTS, it is important to
       use a `step_size` such as 1e-4. If not, you are likely to experience LAPACK
       errors regarding positive-definiteness.

    For example usage, refer to
    `pyro/examples/lkj.py <https://github.com/pyro-ppl/pyro/blob/dev/examples/lkj.py>`_.

    :param int d: Dimensionality of the matrix
    :param torch.Tensor eta: A single positive number parameterizing the distribution.
    """
    arg_constraints = {"eta": constraints.positive}
    support = corr_cholesky_constraint
    has_rsample = False

    def __init__(self, d, eta, validate_args=None):
        if eta.numel() != 1:
            raise ValueError("eta must be a single number; for a larger batch size, call expand")
        if d <= 1:
            raise ValueError("d must be > 1 in any correlation matrix")
        eta = eta.squeeze()
        vector_size = (d * (d - 1)) // 2
        alpha = eta.add(0.5 * (d - 1.0))

        concentrations = torch.empty(vector_size, dtype=eta.dtype, device=eta.device)
        i = 0
        for k in range(d - 1):
            alpha -= .5
            concentrations[..., i:(i + d - k - 1)] = alpha
            i += d - k - 1
        self._gen = Beta(concentrations, concentrations)
        self.eta = eta
        self._d = d
        self._lkj_constant = None
        super().__init__(torch.Size(), torch.Size((d, d)), validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            y = self._gen.sample(sample_shape=sample_shape + self.batch_shape)
        z = y.mul(2).add(-1.0)
        return _vector_to_l_cholesky(z)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCorrCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new._gen = self._gen
        new.eta = self.eta
        new._d = self._d
        new._lkj_constant = self._lkj_constant
        super(LKJCorrCholesky, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def lkj_constant(self, eta, K):
        if self._lkj_constant is not None:
            return self._lkj_constant

        Km1 = K - 1

        constant = torch.lgamma(eta.add(0.5 * Km1)).mul(Km1)

        k = torch.linspace(start=1, end=Km1, steps=Km1, dtype=eta.dtype, device=eta.device)
        constant -= (k.mul(math.log(math.pi) * 0.5) + torch.lgamma(eta.add(0.5 * (Km1 - k)))).sum()

        self._lkj_constant = constant
        return constant

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)

        eta = self.eta

        lp = self.lkj_constant(eta, self._d)

        Km1 = self._d - 1

        log_diagonals = x.diagonal(offset=0, dim1=-1, dim2=-2)[..., 1:].log()
        # TODO: Figure out why the `device` kwarg to torch.linspace seems to not work in certain situations,
        # and a seemingly redundant .to(x.device) is needed below.
        values = log_diagonals * torch.linspace(start=Km1 - 1, end=0, steps=Km1,
                                                dtype=x.dtype,
                                                device=x.device).expand_as(log_diagonals).to(x.device)

        values += log_diagonals.mul(eta.mul(2).add(-2.0))
        values = values.sum(-1) + lp
        values, _ = torch.broadcast_tensors(values, torch.empty(self.batch_shape))
        return values
