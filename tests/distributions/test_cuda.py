from __future__ import absolute_import, division, print_function

import pytest

from tests.common import assert_equal, requires_cuda, tensors_default_to, xfail_if_not_implemented


@requires_cuda
def test_sample(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        with tensors_default_to("cpu"):
            params = dist.get_dist_params(idx)
        try:
            cpu_value = dist.pyro_dist.sample(**params)
        except ValueError as e:
            pytest.xfail('CPU version fails: {}'.format(e))
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with tensors_default_to("cuda"):
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist.sample(**params)
        assert cuda_value.is_cuda

        assert_equal(cpu_value.size(), cuda_value.size())


@requires_cuda
def test_log_pdf(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        with tensors_default_to("cpu"):
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        with xfail_if_not_implemented():
            cpu_value = dist.pyro_dist.log_pdf(data, **params)
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with tensors_default_to("cuda"):
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist.log_pdf(data, **params)
        assert cuda_value.is_cuda

        assert_equal(cpu_value, cuda_value.cpu())


@requires_cuda
def test_batch_log_pdf(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        with tensors_default_to("cpu"):
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        with xfail_if_not_implemented():
            cpu_value = dist.pyro_dist.log_prob(data, **params)
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with tensors_default_to("cuda"):
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist.log_prob(data, **params)
        assert cuda_value.is_cuda

        assert_equal(cpu_value, cuda_value.cpu())
