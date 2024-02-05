# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions.transforms import Transform
from torch.special import expit, logit

from .. import constraints


# This class is a port of https://num.pyro.ai/en/stable/_modules/numpyro/distributions/transforms.html#SimplexToOrderedTransform
class SimplexToOrderedTransform(Transform):
    """
    Transform a simplex into an ordered vector (via difference in Logistic CDF between cutpoints)
    Used in [1] to induce a prior on latent cutpoints via transforming ordered category probabilities.

    :param anchor_point: Anchor point is a nuisance parameter to improve the identifiability of the transform.
        For simplicity, we assume it is a scalar value, but it is broadcastable x.shape[:-1].
        For more details please refer to Section 2.2 in [1]

    **References:**

    1. *Ordinal Regression Case Study, section 2.2*,
       M. Betancourt, https://betanalpha.github.io/assets/case_studies/ordinal_regression.html

    """

    domain = constraints.simplex
    codomain = constraints.ordered_vector

    def __init__(self, anchor_point=None):
        super().__init__()
        self.anchor_point = (
            anchor_point if anchor_point is not None else torch.tensor(0.0)
        )

    def _call(self, x):
        s = torch.cumsum(x[..., :-1], axis=-1)
        y = logit(s) + torch.unsqueeze(self.anchor_point, -1)
        return y

    def _inverse(self, y):
        y = y - torch.unsqueeze(self.anchor_point, -1)
        s = expit(y)
        # x0 = s0, x1 = s1 - s0, x2 = s2 - s1,..., xn = 1 - s[n-1]
        # add two boundary points 0 and 1
        s = torch.concat(
            [torch.zeros_like(s)[..., :1], s, torch.ones_like(s)[..., :1]], dim=-1
        )
        x = s[..., 1:] - s[..., :-1]
        return x

    def log_abs_det_jacobian(self, x, y):
        # |dp/dc| = |dx/dy| = prod(ds/dy) = prod(expit'(y))
        # we know log derivative of expit(y) is `-softplus(y) - softplus(-y)`
        J_logdet = (
            torch.nn.functional.softplus(y) + torch.nn.functional.softplus(-y)
        ).sum(-1)
        return J_logdet

    def __eq__(self, other):
        if not isinstance(other, SimplexToOrderedTransform):
            return False
        return torch.all(torch.equal(self.anchor_point, other.anchor_point))

    def forward_shape(self, shape):
        return shape[:-1] + (shape[-1] - 1,)

    def inverse_shape(self, shape):
        return shape[:-1] + (shape[-1] + 1,)
