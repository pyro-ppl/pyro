# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.autograd import grad

from tests.common import assert_equal, requires_cuda, tensors_default_to, xfail_if_not_implemented


@requires_cuda
def test_sample(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        with tensors_default_to("cpu"):
            params = dist.get_dist_params(idx)
        try:
            with xfail_if_not_implemented():
                cpu_value = dist.pyro_dist(**params).sample()
        except ValueError as e:
            pytest.xfail('CPU version fails: {}'.format(e))
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with tensors_default_to("cuda"):
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist(**params).sample()
        assert cuda_value.is_cuda

        assert_equal(cpu_value.size(), cuda_value.size())


@requires_cuda
def test_rsample(dist):
    if not dist.pyro_dist.has_rsample:
        return
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        with tensors_default_to("cpu"):
            params = dist.get_dist_params(idx)
            grad_params = [key for key, val in params.items()
                           if torch.is_tensor(val) and val.dtype in (torch.float32, torch.float64)]
            for key in grad_params:
                val = params[key].clone()
                val.requires_grad = True
                params[key] = val
        try:
            with xfail_if_not_implemented():
                cpu_value = dist.pyro_dist(**params).rsample()
                cpu_grads = grad(cpu_value.sum(), [params[key] for key in grad_params])
        except ValueError as e:
            pytest.xfail('CPU version fails: {}'.format(e))
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with tensors_default_to("cuda"):
            params = dist.get_dist_params(idx)
            for key in grad_params:
                val = params[key].clone()
                val.requires_grad = True
                params[key] = val
        cuda_value = dist.pyro_dist(**params).rsample()
        assert cuda_value.is_cuda
        assert_equal(cpu_value.size(), cuda_value.size())

        cuda_grads = grad(cuda_value.sum(), [params[key] for key in grad_params])
        for cpu_grad, cuda_grad in zip(cpu_grads, cuda_grads):
            assert_equal(cpu_grad.size(), cuda_grad.size())


@requires_cuda
def test_log_prob(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        with tensors_default_to("cpu"):
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        with xfail_if_not_implemented():
            cpu_value = dist.pyro_dist(**params).log_prob(data)
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with tensors_default_to("cuda"):
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist(**params).log_prob(data)
        assert cuda_value.is_cuda

        assert_equal(cpu_value, cuda_value.cpu())
