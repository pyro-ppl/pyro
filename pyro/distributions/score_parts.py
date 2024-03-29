# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple, Optional, Union

import torch

from pyro.distributions.util import scale_and_mask


class ScoreParts(NamedTuple):
    """
    This data structure stores terms used in stochastic gradient estimators that
    combine the pathwise estimator and the score function estimator.
    """

    log_prob: torch.Tensor
    score_function: torch.Tensor
    entropy_term: torch.Tensor

    def scale_and_mask(
        self,
        scale: Union[float, torch.Tensor] = 1.0,
        mask: Optional[torch.BoolTensor] = None,
    ) -> "ScoreParts":
        """
        Scale and mask appropriate terms of a gradient estimator by a data multiplicity factor.
        Note that the `score_function` term should not be scaled or masked.

        :param scale: a positive scale
        :type scale: torch.Tensor or number
        :param mask: an optional masking tensor
        :type mask: torch.BoolTensor or None
        """
        log_prob = scale_and_mask(self.log_prob, scale, mask)
        score_function = self.score_function  # not scaled
        entropy_term = scale_and_mask(self.entropy_term, scale, mask)
        return ScoreParts(log_prob, score_function, entropy_term)
