import pytest

from tests.common import assert_equal, cuda_tensors, requires_cuda, xfail_if_not_implemented


@requires_cuda
def test_sample(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        params = dist.get_dist_params(idx)
        try:
            cpu_value = dist.pyro_dist.sample(**params)
        except ValueError as e:
            pytest.xfail('CPU version fails: {}'.format(e))
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with cuda_tensors():
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist.sample(**params)
        assert cuda_value.is_cuda

        assert_equal(cpu_value.size(), cuda_value.size())


@requires_cuda
def test_log_pdf(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        data = dist.get_test_data(idx)
        params = dist.get_dist_params(idx)
        cpu_value = dist.pyro_dist.log_pdf(data, **params)
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with cuda_tensors():
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist.log_pdf(data, **params)
        assert cuda_value.is_cuda

        assert_equal(cpu_value, cuda_value.cpu())


@requires_cuda
def test_batch_log_pdf(dist):
    for idx in range(len(dist.dist_params)):

        # Compute CPU value.
        data = dist.get_test_data(idx)
        params = dist.get_dist_params(idx)
        with xfail_if_not_implemented():
            cpu_value = dist.pyro_dist.batch_log_pdf(data, **params)
        assert not cpu_value.is_cuda

        # Compute GPU value.
        with cuda_tensors():
            data = dist.get_test_data(idx)
            params = dist.get_dist_params(idx)
        cuda_value = dist.pyro_dist.batch_log_pdf(data, **params)
        assert cuda_value.is_cuda

        assert_equal(cpu_value, cuda_value.cpu())
