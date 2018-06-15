from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.torch_distribution import ReshapedDistribution
from pyro.distributions.util import broadcast_shape
from tests.common import assert_equal, xfail_if_not_implemented


def _log_prob_shape(dist, x_size=torch.Size()):
    event_dims = len(dist.event_shape)
    expected_shape = broadcast_shape(dist.shape(), x_size, strict=True)
    if event_dims > 0:
        expected_shape = expected_shape[:-event_dims]
    return expected_shape

# Distribution tests - all distributions


def test_batch_log_prob(dist):
    if dist.scipy_arg_fn is None:
        pytest.skip('{}.log_prob_sum has no scipy equivalent'.format(dist.pyro_dist.__name__))
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        d = dist.pyro_dist(**dist_params)
        test_data = dist.get_test_data(idx)
        log_prob_sum_pyro = d.log_prob(test_data).sum().item()
        log_prob_sum_np = np.sum(dist.get_scipy_batch_logpdf(-1))
        assert_equal(log_prob_sum_pyro, log_prob_sum_np)


def test_batch_log_prob_shape(dist):
    for idx in range(dist.get_num_test_data()):
        dist_params = dist.get_dist_params(idx)
        d = dist.pyro_dist(**dist_params)
        x = dist.get_test_data(idx)
        with xfail_if_not_implemented():
            # Get log_prob shape after broadcasting.
            expected_shape = _log_prob_shape(d, x.size())
            log_p_obj = d.log_prob(x)
            assert log_p_obj.size() == expected_shape


def test_score_errors_event_dim_mismatch(dist):
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        d = dist.pyro_dist(**dist_params)
        test_data_wrong_dims = torch.ones(d.shape() + (1,))
        if len(d.event_shape) > 0:
            if dist.get_test_distribution_name() == 'MultivariateNormal':
                pytest.skip('MultivariateNormal does not do shape validation in log_prob.')
            if dist.get_test_distribution_name() == 'LowRankMultivariateNormal':
                pytest.skip('LowRankMultivariateNormal does not do shape validation in log_prob.')
            with pytest.raises((ValueError, RuntimeError)):
                d.log_prob(test_data_wrong_dims)


def test_score_errors_non_broadcastable_data_shape(dist):
    for idx in dist.get_batch_data_indices():
        dist_params = dist.get_dist_params(idx)
        d = dist.pyro_dist(**dist_params)
        shape = d.shape()
        non_broadcastable_shape = (shape[0] + 1,) + shape[1:]
        test_data_non_broadcastable = torch.ones(non_broadcastable_shape)
        with pytest.raises((ValueError, RuntimeError)):
            d.log_prob(test_data_non_broadcastable)


# Distributions tests - discrete distributions

def test_enumerate_support(discrete_dist):
    expected_support = discrete_dist.expected_support
    expected_support_non_vec = discrete_dist.expected_support_non_vec
    if not expected_support:
        pytest.skip("enumerate_support not tested for distribution")
    Dist = discrete_dist.pyro_dist
    actual_support_non_vec = Dist(**discrete_dist.get_dist_params(0)).enumerate_support()
    actual_support = Dist(**discrete_dist.get_dist_params(-1)).enumerate_support()
    assert_equal(actual_support.data, torch.tensor(expected_support))
    assert_equal(actual_support_non_vec.data, torch.tensor(expected_support_non_vec))


@pytest.mark.parametrize("dist_class, args", [
    (dist.Normal, {"loc": torch.tensor(0.0), "scale": torch.tensor(-1.0)}),
    (dist.Gamma, {"concentration": -1.0, "rate": 1.0}),
    (dist.Exponential, {"rate": -2})
])
@pytest.mark.parametrize("validate_args", [True, False])
def test_distribution_validate_args(dist_class, args, validate_args):
    with pyro.validation_enabled(validate_args):
        if not validate_args:
            dist_class(**args)
        else:
            with pytest.raises(ValueError):
                dist_class(**args)


def check_sample_shapes(small, large):
    if isinstance(small, dist.LogNormal) or (
            isinstance(small, ReshapedDistribution) and
            isinstance(small.base_dist, dist.LogNormal)):
        # Ignore broadcasting bug in LogNormal:
        # https://github.com/pytorch/pytorch/pull/7269
        return
    try:
        x = small.sample()
        assert_equal(small.log_prob(x).expand(large.batch_shape), large.log_prob(x))
        x = large.sample()
        assert_equal(small.log_prob(x), large.log_prob(x))
    except NotImplementedError:
        pass


@pytest.mark.parametrize('sample_shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('shape_type', [torch.Size, tuple, list])
def test_expand_by(dist, sample_shape, shape_type):
    for idx in range(dist.get_num_test_data()):
        small = dist.pyro_dist(**dist.get_dist_params(idx))
        large = small.expand_by(shape_type(sample_shape))
        assert large.batch_shape == sample_shape + small.batch_shape
        check_sample_shapes(small, large)


@pytest.mark.parametrize('sample_shape', [(), (2,), (2, 3)])
@pytest.mark.parametrize('shape_type', [torch.Size, tuple, list])
def test_expand_new_dim(dist, sample_shape, shape_type):
    for idx in range(dist.get_num_test_data()):
        small = dist.pyro_dist(**dist.get_dist_params(idx))
        large = small.expand(shape_type(sample_shape + small.batch_shape))
        assert large.batch_shape == sample_shape + small.batch_shape
        check_sample_shapes(small, large)


@pytest.mark.parametrize('shape_type', [torch.Size, tuple, list])
def test_expand_existing_dim(dist, shape_type):
    for idx in range(dist.get_num_test_data()):
        small = dist.pyro_dist(**dist.get_dist_params(idx))
        for dim, size in enumerate(small.batch_shape):
            if size != 1:
                continue
            batch_shape = list(small.batch_shape)
            batch_shape[dim] = 5
            batch_shape = torch.Size(batch_shape)
            with xfail_if_not_implemented():
                large = small.expand(shape_type(batch_shape))
            assert large.batch_shape == batch_shape
            check_sample_shapes(small, large)


@pytest.mark.parametrize("sample_shapes", [
    [(2, 1), (2, 3)],
    [(2, 1, 1), (2, 1, 3), (2, 5, 3)],
])
def test_subsequent_expands_ok(dist, sample_shapes):
    for idx in range(dist.get_num_test_data()):
        d = dist.pyro_dist(**dist.get_dist_params(idx))
        original_batch_shape = d.batch_shape
        for shape in sample_shapes:
            proposed_batch_shape = torch.Size(shape) + original_batch_shape
            with xfail_if_not_implemented():
                n = d.expand(proposed_batch_shape)
            assert n.batch_shape == proposed_batch_shape
            check_sample_shapes(d, n)
            d = n


@pytest.mark.parametrize("initial_shape, proposed_shape", [
    [(2, 1), (4, 3)],
    [(2, 4), (2, 2, 1)],
    [(1, 2, 1), (2, 1)],
])
def test_expand_error(dist, initial_shape, proposed_shape):
    for idx in range(dist.get_num_test_data()):
        small = dist.pyro_dist(**dist.get_dist_params(idx))
        large = small.expand(torch.Size(initial_shape) + small.batch_shape)
        proposed_batch_shape = torch.Size(proposed_shape) + small.batch_shape
        with pytest.raises(ValueError):
            large.expand(proposed_batch_shape)


@pytest.mark.parametrize("extra_event_dims,expand_shape", [
    (0, [4, 3, 2, 1]),
    (0, [4, 3, 2, 2]),
    (1, [5, 4, 3, 2]),
    (2, [5, 4, 3]),
])
def test_expand_reshaped_distribution(extra_event_dims, expand_shape):
    probs = torch.ones(1, 6) / 6
    d = dist.OneHotCategorical(probs)
    reshaped_dist = d.expand_by([4, 1, 1]).independent(extra_event_dims)
    full_shape = torch.Size([4, 1, 1, 1, 6])
    cut = 4 - extra_event_dims
    batch_shape, event_shape = full_shape[:cut], full_shape[cut:]
    assert reshaped_dist.batch_shape == batch_shape
    assert reshaped_dist.event_shape == event_shape
    large = reshaped_dist.expand(expand_shape)
    assert large.batch_shape == torch.Size(expand_shape)
    assert large.event_shape == torch.Size(event_shape)

    # Change base_dist only if sample_shape cannot be adjusted.
    if extra_event_dims >= 1:
        assert large.base_dist == reshaped_dist.base_dist
    else:
        if expand_shape[-1] == 1:
            assert large.base_dist == reshaped_dist.base_dist
        else:
            assert large.base_dist.batch_shape == torch.Size(expand_shape)

    # Throws error when batch shape cannot be broadcasted
    with pytest.raises(ValueError):
        reshaped_dist.expand(expand_shape + [3])

    # Throws error when trying to shrink existing batch shape
    with pytest.raises(ValueError):
        large.expand(expand_shape[1:])
