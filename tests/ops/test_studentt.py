import math

import pytest
import torch

import pyro.distributions as dist
from pyro.distributions.util import broadcast_shape
from pyro.ops.studentt import (
    StudentT,
    studentt_tensordot,
    matrix_and_mvt_to_studentt,
    mvt_to_studentt,
    _moment_matching,
)
from tests.common import assert_close
from tests.ops.random import assert_close_studentt, random_studentt, random_mvt


@pytest.mark.parametrize("extra_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("log_normalizer_shape,info_vec_shape,precision_shape,alpha_shape,beta_shape", [
    ((), (), (), (), ()),
    ((5,), (), (), (), ()),
    ((), (5,), (), (), ()),
    ((), (), (5,), (), ()),
    ((), (), (), (5,), ()),
    ((), (), (), (), (5,)),
    ((3, 1, 1), (1, 4, 1), (1, 1, 5), (3, 4, 1), (1, 4, 5)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_expand(extra_shape, log_normalizer_shape, info_vec_shape, precision_shape, alpha_shape, beta_shape, dim):
    rank = dim + dim
    log_normalizer = torch.randn(log_normalizer_shape)
    info_vec = torch.randn(info_vec_shape + (dim,))
    precision = torch.randn(precision_shape + (dim, rank))
    precision = precision.matmul(precision.transpose(-1, -2))
    alpha = torch.randn(alpha_shape).exp()
    beta = torch.randn(beta_shape).exp()
    studentt = StudentT(log_normalizer, info_vec, precision, alpha, beta)

    expected_shape = extra_shape + broadcast_shape(
        log_normalizer_shape, info_vec_shape, precision_shape, alpha_shape, beta_shape)
    actual = studentt.expand(expected_shape)
    assert actual.batch_shape == expected_shape


@pytest.mark.parametrize("old_shape,new_shape", [
    ((6,), (3, 2)),
    ((5, 6), (5, 3, 2)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_reshape(old_shape, new_shape, dim):
    studentt = random_studentt(old_shape, dim)

    # reshape to new
    new = studentt.reshape(new_shape)
    assert new.batch_shape == new_shape

    # reshape back to old
    st = new.reshape(old_shape)
    assert_close_studentt(st, studentt)


@pytest.mark.parametrize("shape,cat_dim,split", [
    ((4, 7, 6), -1, (2, 1, 3)),
    ((4, 7, 6), -2, (1, 1, 2, 3)),
    ((4, 7, 6), 1, (1, 1, 2, 3)),
], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_cat(shape, cat_dim, split, dim):
    assert sum(split) == shape[cat_dim]
    studentt = random_studentt(shape, dim)
    parts = []
    end = 0
    for size in split:
        beg, end = end, end + size
        if cat_dim == -1:
            part = studentt[..., beg: end]
        elif cat_dim == -2:
            part = studentt[..., beg: end, :]
        elif cat_dim == 1:
            part = studentt[:, beg: end]
        else:
            raise ValueError
        parts.append(part)

    actual = StudentT.cat(parts, cat_dim)
    assert_close_studentt(actual, studentt)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("left", [0, 1, 2])
@pytest.mark.parametrize("right", [0, 1, 2])
def test_pad(shape, left, right, dim):
    expected = random_studentt(shape, dim)
    padded = expected.event_pad(left=left, right=right)
    assert padded.batch_shape == expected.batch_shape
    assert padded.dim() == left + expected.dim() + right
    mid = slice(left, padded.dim() - right)
    assert_close(padded.joint.info_vec[..., mid], expected.joint.info_vec)
    assert_close(padded.joint.precision[..., mid, mid], expected.joint.precision)
    assert_close(padded.joint.alpha, expected.joint.alpha)
    assert_close(padded.joint.beta, expected.joint.beta)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize_shape(batch_shape, left, right):
    dim = left + right
    st = random_studentt(batch_shape, dim)
    assert st.marginalize(left=left).dim() == right
    assert st.marginalize(right=right).dim() == left


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize(batch_shape, left, right):
    dim = left + right
    st = random_studentt(batch_shape, dim)
    assert_close(st.marginalize(left=left).event_logsumexp(),
                 st.event_logsumexp())
    assert_close(st.marginalize(right=right).event_logsumexp(),
                 st.event_logsumexp())


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_marginalize_condition(sample_shape, batch_shape, left, right):
    dim = left + right
    st = random_studentt(batch_shape, dim)
    x = torch.randn(sample_shape + (1,) * len(batch_shape) + (right,))
    assert_close(st.marginalize(left=left).log_density(x),
                 st.condition(x).event_logsumexp())


@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_conditional_mvt(left, right):
    # Ref: http://users.isy.liu.se/en/rt/roth/student.pdf section 5
    dim = left + right
    mvt = random_mvt((), dim)
    st = mvt_to_studentt(mvt)
    value = torch.randn((dim,))
    right_value = value[left:]
    conditioned = st.condition(right_value)

    sigma = mvt.covariance_matrix
    sigma22 = sigma[left:, left:]
    sigma22_inv = sigma22.inverse()
    sigma22_inv_x2_minus_mu2 = sigma22_inv.matmul((right_value - mvt.loc[left:]).unsqueeze(-1)).squeeze(-1)
    mu1_2 = mvt.loc[:left] + sigma[:left, left:].matmul(sigma22_inv_x2_minus_mu2)
    cond_sigma = sigma[:left, :left] - sigma[:left, left:].matmul(sigma22_inv).matmul(sigma[left:, :left])
    cond_coef = (mvt.df + sigma22_inv_x2_minus_mu2.mul(right_value - mvt.loc[left:]).sum(-1)) / (mvt.df + right)
    scale_tril = (cond_coef.unsqueeze(-1).unsqueeze(-1) * cond_sigma).cholesky()
    expected_cond_mvt = dist.MultivariateStudentT(mvt.df + right, mu1_2, scale_tril)
    expected_cond_st = mvt_to_studentt(expected_cond_mvt)
    # p(a | b) = p(a, b) / p(b)
    p_b = dist.MultivariateStudentT(mvt.df, mvt.loc[left:], sigma22.cholesky()).log_prob(right_value)
    expected_cond_st.joint.log_normalizer += p_b
    assert_close_studentt(conditioned, expected_cond_st)


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("left", [1, 2, 3])
@pytest.mark.parametrize("right", [1, 2, 3])
def test_condition(sample_shape, batch_shape, left, right):
    dim = left + right
    st = random_studentt(batch_shape, dim)
    st.joint.precision += torch.eye(dim) * 0.1
    value = torch.randn(sample_shape + (1,) * len(batch_shape) + (dim,))
    left_value, right_value = value[..., :left], value[..., left:]

    conditioned = st.condition(right_value)
    assert conditioned.batch_shape == sample_shape + st.batch_shape
    assert conditioned.dim() == left

    actual = conditioned.log_density(left_value)
    expected = st.log_density(value)
    assert_close(actual, expected)


@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_logsumexp(batch_shape, dim):
    # force df > 2: not so heavy tail
    st = random_studentt(batch_shape, dim, df_min=2.)
    st.joint.info_vec *= 0.1  # approximately centered
    st.joint.precision += torch.eye(dim) * 0.1

    num_samples = 200000
    scale = 20
    samples = torch.rand((num_samples,) + (1,) * len(batch_shape) + (dim,)) * scale - scale / 2
    expected = st.log_density(samples).logsumexp(0) + math.log(scale ** dim / num_samples)
    actual = st.event_logsumexp()
    assert_close(actual, expected, atol=0.06, rtol=0.06)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_mvt_to_studentt(sample_shape, batch_shape, dim):
    mvt = random_mvt(batch_shape, dim)
    studentt = mvt_to_studentt(mvt)
    value = mvt.sample(sample_shape)
    actual_log_prob = studentt.log_density(value)
    expected_log_prob = mvt.log_prob(value)
    assert_close(actual_log_prob, expected_log_prob, rtol=0.1)


@pytest.mark.parametrize("sample_shape", [(), (7,), (6, 5)], ids=str)
@pytest.mark.parametrize("batch_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("x_dim", [1, 2, 3])
@pytest.mark.parametrize("y_dim", [1, 2, 3])
def test_matrix_and_mvt_to_studentt(sample_shape, batch_shape, x_dim, y_dim):
    matrix = torch.randn(batch_shape + (x_dim, y_dim))
    y_mvt = random_mvt(batch_shape, y_dim)
    st = matrix_and_mvt_to_studentt(matrix, y_mvt)
    xy = torch.randn(sample_shape + (1,) * len(batch_shape) + (x_dim + y_dim,))
    x, y = xy[..., :x_dim], xy[..., x_dim:]
    y_pred = x.unsqueeze(-2).matmul(matrix).squeeze(-2)
    actual_log_prob = st.log_density(xy)
    expected_log_prob = y_mvt.log_prob(y - y_pred)
    assert_close(actual_log_prob, expected_log_prob)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_moment_matching(shape, dim, order):
    x_mvt = random_mvt(shape, dim, df_min=order + 3)
    x = mvt_to_studentt(x_mvt)
    new_df = torch.randn(shape).exp() + order + 1
    y = _moment_matching(x, new_df, order)
    # assert y is a normalized density
    assert_close(y.event_logsumexp(), torch.zeros(shape))
    y_mvt = y.to_mvt()
    assert_close(x_mvt.loc, y_mvt.loc)
    n = 100000
    x_samples = x_mvt.sample(torch.Size([n]))
    y_samples = y_mvt.sample(torch.Size([n]))
    absolute_mm_x = (x_samples - x_mvt.loc).abs().pow(order / dim).sum(-1).mean(0)
    absolute_mm_y = (y_samples - y_mvt.loc).abs().pow(order / dim).sum(-1).mean(0)
    assert_close(absolute_mm_x, absolute_mm_y, atol=0.3 * order)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_add(shape, dim, order):
    x_mvt = random_mvt(shape, dim, df_min=order + 3)
    x = mvt_to_studentt(x_mvt)
    y_mvt = random_mvt(shape, dim, df_min=order + 1)
    y = mvt_to_studentt(y_mvt)
    xy = x.event_pad(right=dim).add(y.event_pad(left=dim), order)
    # assert xy is a normalized density
    assert_close(xy.event_logsumexp(), torch.zeros(shape))
    # assert tail is preserved
    xy_mvt = xy.to_mvt()
    assert_close(xy_mvt.df, torch.min(x_mvt.df, y_mvt.df))
    # assert moment matching
    assert_close(xy_mvt.loc, torch.cat([x_mvt.loc, y_mvt.loc], -1))
    n = 100000
    x_samples0 = x_mvt.sample(torch.Size([n]))
    x_samples1 = xy.marginalize(right=dim).to_mvt().sample(torch.Size([n]))
    absolute_mm_0 = (x_samples0 - x_mvt.loc).abs().pow(order / dim).sum(-1).mean(0)
    absolute_mm_1 = (x_samples1 - x_mvt.loc).abs().pow(order / dim).sum(-1).mean(0)
    assert_close(absolute_mm_0, absolute_mm_1, atol=0.3 * order)


@pytest.mark.parametrize("x_batch_shape,y_batch_shape", [
    ((), ()),
    ((3,), ()),
    ((), (3,)),
    ((2, 1), (3,)),
    ((2, 3), (2, 3,)),
], ids=str)
@pytest.mark.parametrize("x_dim,y_dim,dot_dims", [
    (0, 0, 0),
    (0, 2, 0),
    (1, 0, 0),
    (2, 1, 0),
    (3, 3, 3),
    (3, 2, 1),
    (3, 2, 2),
    (5, 4, 2),
], ids=str)
@pytest.mark.parametrize("x_rank,y_rank", [
    (1, 1), (4, 1), (1, 4), (4, 4)
], ids=str)
def test_studentt_tensordot(dot_dims,
                            x_batch_shape, x_dim, x_rank,
                            y_batch_shape, y_dim, y_rank):
    x_rank = min(x_rank, x_dim)
    y_rank = min(y_rank, y_dim)
    x = random_studentt(x_batch_shape, x_dim, x_rank, df_min=1.)
    y = random_studentt(y_batch_shape, y_dim, y_rank, df_min=1.)
    y.mask[:dot_dims].fill_(False)
    try:
        z = studentt_tensordot(x, y, dot_dims)
    except RuntimeError:
        pytest.skip("Cannot marginalize the common variables of two StudentTs.")

    assert z.batch_shape == broadcast_shape(x_batch_shape, y_batch_shape)
    assert z.dim() == x_dim + y_dim - 2 * dot_dims


@pytest.mark.parametrize("x_dim", [1, 2, 3])
@pytest.mark.parametrize("y_dim", [1, 2, 3])
def test_filter_prediction(x_dim, y_dim):
    # y = x @ M + e, where y is the latent variable of the next time step
    x_mvt = random_mvt((), x_dim, df_min=1.)
    matrix = torch.randn((x_dim, y_dim))
    e_mvt = random_mvt((), y_dim, df_min=1.)
    x = mvt_to_studentt(x_mvt)
    e = mvt_to_studentt(e_mvt)
    xe = studentt_tensordot(x, e)
    xe_mvt = xe.to_mvt()
    matrix_ext = torch.eye(x_dim + y_dim)
    matrix_ext[x_dim:, :x_dim] = matrix.transpose(-2, -1)
    expected_df = xe_mvt.df
    expected_loc = matrix_ext.matmul(xe_mvt.loc.unsqueeze(-1)).squeeze(-1)
    expected_scale_tril = matrix_ext.matmul(xe_mvt.scale_tril)

    y = matrix_and_mvt_to_studentt(matrix, e_mvt)
    xy = x.event_pad(right=y_dim) + y
    xy_mvt = xy.to_mvt()
    assert_close(xy_mvt.df, expected_df)
    assert_close(xy_mvt.loc, expected_loc)
    assert_close(xy_mvt.scale_tril, expected_scale_tril)


@pytest.mark.parametrize("x_dim", [1, 2, 3])
@pytest.mark.parametrize("y_dim", [1, 2, 3])
def test_filter_update(x_dim, y_dim):
    pass
