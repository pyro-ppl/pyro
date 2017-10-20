from tests.common import assert_equal, cuda_tensors, requires_cuda, xfail_if_not_implemented

SINGLE_TEST_DATUM_IDX = 0
BATCH_TEST_DATA_IDX = -1


@requires_cuda
def test_sample(dist):
    idx = SINGLE_TEST_DATUM_IDX

    # Compute CPU value.
    params = dist.get_dist_params(idx)
    cpu_value = dist.pyro_dist.sample(**params)
    assert not cpu_value.is_cuda

    # Compute GPU value.
    with cuda_tensors():
        params = dist.get_dist_params(idx)
    cuda_value = dist.pyro_dist.sample(**params)
    assert cuda_value.is_cuda

    assert_equal(cpu_value.size(), cuda_value.size())


@requires_cuda
def test_log_pdf(dist):
    idx = SINGLE_TEST_DATUM_IDX

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
    idx = BATCH_TEST_DATA_IDX

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
