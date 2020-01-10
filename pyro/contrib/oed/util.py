# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from pyro.contrib.util import get_indices
from pyro.contrib.oed.glmm import analytic_posterior_cov
from pyro.infer.autoguide.utils import mean_field_entropy


def linear_model_ground_truth(model, design, observation_labels, target_labels, eig=True):
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    w_sd = torch.cat(list(model.w_sds.values()), dim=-1)
    prior_cov = torch.diag(w_sd**2)
    design_shape = design.shape
    posterior_covs = [analytic_posterior_cov(prior_cov, x, model.obs_sd) for x in
                      torch.unbind(design.reshape(-1, design_shape[-2], design_shape[-1]))]
    target_indices = get_indices(target_labels, tensors=model.w_sds)
    target_posterior_covs = [S[target_indices, :][:, target_indices] for S in posterior_covs]
    output = torch.tensor([0.5 * torch.logdet(2 * math.pi * math.e * C)
                           for C in target_posterior_covs])
    if eig:
        prior_entropy = mean_field_entropy(model, [design], whitelist=target_labels)
        output = prior_entropy - output

    return output.reshape(design.shape[:-2])
