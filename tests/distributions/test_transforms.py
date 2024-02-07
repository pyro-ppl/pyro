# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import operator
from functools import partial, reduce
from unittest import TestCase

import pytest
import torch

import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.distributions import constraints
from tests.common import assert_close

pytestmark = pytest.mark.init(rng_seed=123)


class Flatten(dist.TransformModule):
    """
    Used to handle transforms with `event_dim > 1` until we have a Reshape transform in PyTorch
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector

    def __init__(self, transform, input_shape):
        super().__init__(cache_size=1)
        assert transform.domain.event_dim == len(input_shape)
        output_shape = transform.forward_shape(input_shape)
        assert len(output_shape) >= transform.codomain.event_dim
        output_shape = output_shape[len(output_shape) - transform.codomain.event_dim :]

        self.transform = transform
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _call(self, x):
        x = x.reshape(x.shape[:-1] + self.input_shape)
        y = self.transform._call(x)
        y = y.reshape(y.shape[: y.dim() - len(self.output_shape)] + (-1,))
        return y

    def _inverse(self, y):
        y = y.reshape(y.shape[:-1] + self.output_shape)
        x = self.transform._inverse(y)
        x = x.reshape(x.shape[: x.dim() - len(self.input_shape)] + (-1,))
        return x

    def log_abs_det_jacobian(self, x, y):
        x = x.reshape(x.shape[:-1] + self.input_shape)
        y = y.reshape(y.shape[:-1] + self.output_shape)
        return self.transform.log_abs_det_jacobian(x, y)

    def parameters(self):
        return self.transform.parameters()


class TransformTests(TestCase):
    def setUp(self):
        # Epsilon is used to compare numerical gradient to analytical one
        self.epsilon = 1e-4

        # Delta is tolerance for testing inverse, f(f^{-1}(x)) = x
        self.delta = 1e-6

    def _test_jacobian(self, input_dim, transform):
        jacobian = torch.zeros(input_dim, input_dim)

        def nonzero(x):
            return torch.sign(torch.abs(x))

        x = torch.randn(1, input_dim)
        y = transform(x)
        if transform.event_dim == 1:
            analytic_ldt = transform.log_abs_det_jacobian(x, y).data
        else:
            analytic_ldt = transform.log_abs_det_jacobian(x, y).sum(-1).data

        for j in range(input_dim):
            for k in range(input_dim):
                epsilon_vector = torch.zeros(1, input_dim)
                epsilon_vector[0, j] = self.epsilon
                delta = (
                    transform(x + 0.5 * epsilon_vector)
                    - transform(x - 0.5 * epsilon_vector)
                ) / self.epsilon
                jacobian[j, k] = float(delta[0, k].data.sum())

        # Apply permutation for autoregressive flows with a network
        if hasattr(transform, "arn") and "get_permutation" in dir(transform.arn):
            permutation = transform.arn.get_permutation()
            permuted_jacobian = jacobian.clone()
            for j in range(input_dim):
                for k in range(input_dim):
                    permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]
            jacobian = permuted_jacobian

        # For autoregressive flow, Jacobian is sum of diagonal, otherwise need full determinate
        if hasattr(transform, "autoregressive") and transform.autoregressive:
            numeric_ldt = torch.sum(torch.log(torch.diag(jacobian)))
        else:
            numeric_ldt = torch.log(torch.abs(jacobian.det()))

        ldt_discrepancy = (analytic_ldt - numeric_ldt).abs()
        assert ldt_discrepancy < self.epsilon

        # Test that lower triangular with unit diagonal for autoregressive flows
        if hasattr(transform, "autoregressive"):
            diag_sum = torch.sum(torch.diag(nonzero(jacobian)))
            lower_sum = torch.sum(torch.tril(nonzero(jacobian), diagonal=-1))
            assert diag_sum == float(input_dim)
            assert lower_sum == float(0.0)

    def _test_inverse(self, shape, transform, base_dist_type="normal"):
        # Test g^{-1}(g(x)) = x
        # NOTE: Calling _call and _inverse directly bypasses caching
        if base_dist_type == "dirichlet":
            base_dist = dist.Dirichlet(torch.ones(shape) * 10.0)
        elif base_dist_type == "normal":
            base_dist = dist.Normal(torch.zeros(shape), torch.ones(shape))
        else:
            raise ValueError(f"Unknown base distribution type: {base_dist_type}")
        x_true = base_dist.sample(torch.Size([10]))
        y = transform._call(x_true)
        x_calculated = transform._inverse(y)
        if transform.bijective:
            assert (x_true - x_calculated).abs().max().item() < self.delta
        else:
            # Check the weaker pseudoinverse equation.
            y2 = transform._call(x_calculated)
            assert (y - y2).abs().max().item() < self.delta

        if transform.bijective:
            # Test that Jacobian after inverse op is same as after forward
            J_1 = transform.log_abs_det_jacobian(x_true, y)
            J_2 = transform.log_abs_det_jacobian(x_calculated, y)
            assert (J_1 - J_2).abs().max().item() < self.delta

    def _test_shape(self, base_shape, transform, base_dist_type="normal"):
        if base_dist_type == "dirichlet":
            base_dist = dist.Dirichlet(torch.ones(base_shape) * 10.0)
        elif base_dist_type == "normal":
            base_dist = dist.Normal(torch.zeros(base_shape), torch.ones(base_shape))
        else:
            raise ValueError(f"Unknown base distribution type: {base_dist_type}")
        sample = dist.TransformedDistribution(base_dist, [transform]).sample()
        assert sample.shape == base_shape

        batch_shape = base_shape[: len(base_shape) - transform.domain.event_dim]
        input_event_shape = base_shape[len(base_shape) - transform.domain.event_dim :]
        output_event_shape = base_shape[
            len(base_shape) - transform.codomain.event_dim :
        ]
        output_shape = batch_shape + output_event_shape
        assert transform.forward_shape(input_event_shape) == output_event_shape
        assert transform.forward_shape(base_shape) == output_shape
        assert transform.inverse_shape(output_event_shape) == input_event_shape
        assert transform.inverse_shape(output_shape) == base_shape

    def _test_autodiff(
        self, input_dim, transform, inverse=False, base_dist_type="normal"
    ):
        """
        This method essentially tests whether autodiff will not throw any errors
        when you're doing maximum-likelihood learning with the transform. Many
        transforms have only one direction with an explicit inverse, hence why we
        pass in the inverse flag.
        """
        temp_transform = transform
        if inverse:
            transform = transform.inv

        if base_dist_type == "dirichlet":
            base_dist = dist.Dirichlet(torch.ones(input_dim) * 10.0)
        elif base_dist_type == "normal":
            base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim))
        else:
            raise ValueError(f"Unknown base distribution type: {base_dist_type}")
        flow_dist = dist.TransformedDistribution(base_dist, [transform])
        optimizer = torch.optim.Adam(temp_transform.parameters())
        x = torch.rand(1, input_dim)
        for _ in range(3):
            optimizer.zero_grad()
            loss = -flow_dist.log_prob(x.detach()).mean()
            loss.backward()
            optimizer.step()

    def _test(
        self,
        transform_factory,
        shape=True,
        jacobian=True,
        inverse=True,
        autodiff=True,
        event_dim=1,
        base_dist_type="normal",
    ):
        for event_shape in [(2,), (5,)]:
            if event_dim > 1:
                event_shape = tuple([event_shape[0] + i for i in range(event_dim)])
            transform = transform_factory(
                event_shape[0] if len(event_shape) == 1 else event_shape
            )

            if inverse:
                self._test_inverse(
                    event_shape, transform, base_dist_type=base_dist_type
                )
            if shape:
                for shape in [(3,), (3, 4), (3, 4, 5)]:
                    base_shape = shape + event_shape
                    self._test_shape(
                        base_shape, transform, base_dist_type=base_dist_type
                    )
            if jacobian and transform.bijective:
                if event_dim > 1:
                    transform = Flatten(transform, event_shape)
                self._test_jacobian(reduce(operator.mul, event_shape, 1), transform)
            if isinstance(transform, dist.TransformModule) and autodiff:
                # If the function doesn't have an explicit inverse, then use the forward op for autodiff
                self._test_autodiff(
                    reduce(operator.mul, event_shape, 1),
                    transform,
                    inverse=not inverse,
                    base_dist_type=base_dist_type,
                )

    def _test_conditional(
        self, conditional_transform_factory, context_dim=3, event_dim=1, **kwargs
    ):
        def transform_factory(input_dim, context_dim=context_dim):
            z = torch.rand(1, context_dim)
            cond_transform = conditional_transform_factory(input_dim, context_dim)
            transform = cond_transform.condition(z)

            # A bit of a hack since conditioned transforms don't expose .parameters()
            transform.parameters = lambda: cond_transform.parameters()

            return transform

        self._test(transform_factory, event_dim=event_dim, **kwargs)

    def test_affine_autoregressive(self):
        for stable in [True, False]:
            self._test(partial(T.affine_autoregressive, stable=stable))

    def test_affine_coupling(self):
        for dim in [-1, -2]:
            self._test(partial(T.affine_coupling, dim=dim), event_dim=-dim)

    def test_batchnorm(self):
        # Need to make moving average statistics non-zeros/ones and set to eval so inverse is valid
        # (see the docs about the differing behaviour of BatchNorm in train and eval modes)
        def transform_factory(input_dim):
            transform = T.batchnorm(input_dim)
            transform._inverse(
                torch.normal(
                    torch.arange(0.0, input_dim),
                    torch.arange(1.0, 1.0 + input_dim) / input_dim,
                )
            )
            transform.eval()
            return transform

        self._test(transform_factory)

    def test_block_autoregressive_jacobians(self):
        for activation in ["ELU", "LeakyReLU", "sigmoid", "tanh"]:
            self._test(
                partial(T.block_autoregressive, activation=activation), inverse=False
            )

        for residual in [None, "normal", "gated"]:
            self._test(
                partial(T.block_autoregressive, residual=residual), inverse=False
            )

    def test_conditional_affine_autoregressive(self):
        self._test_conditional(T.conditional_affine_autoregressive)

    def test_conditional_affine_coupling(self):
        for dim in [-1, -2]:
            self._test_conditional(
                partial(T.conditional_affine_coupling, dim=dim), event_dim=-dim
            )

    def test_conditional_generalized_channel_permute(self, context_dim=3):
        for shape in [(3, 16, 16), (1, 3, 32, 32), (2, 5, 3, 64, 64)]:
            # NOTE: Without changing the interface to generalized_channel_permute I can't reuse general
            # test for `event_dim > 1` transforms
            z = torch.rand(context_dim)
            transform = T.conditional_generalized_channel_permute(
                context_dim=3, channels=shape[-3]
            ).condition(z)
            self._test_shape(shape, transform)
            self._test_inverse(shape, transform)

        for width_dim in [2, 4, 6]:
            input_dim = (width_dim**2) * 3
            self._test_jacobian(
                input_dim, Flatten(transform, (3, width_dim, width_dim))
            )

    def test_conditional_householder(self):
        self._test_conditional(T.conditional_householder)
        self._test_conditional(partial(T.conditional_householder, count_transforms=2))

    def test_conditional_matrix_exponential(self):
        self._test_conditional(T.conditional_matrix_exponential)

    def test_conditional_neural_autoregressive(self):
        self._test_conditional(T.conditional_neural_autoregressive, inverse=False)

    def test_conditional_planar(self):
        self._test_conditional(T.conditional_planar, inverse=False)

    def test_conditional_radial(self):
        self._test_conditional(T.conditional_radial, inverse=False)

    def test_conditional_spline(self):
        for order in ["linear", "quadratic"]:
            self._test_conditional(partial(T.conditional_spline, order=order))

    def test_conditional_spline_autoregressive(self):
        self._test_conditional(T.conditional_spline_autoregressive)

    def test_discrete_cosine(self):
        # NOTE: Need following since helper function unimplemented
        for smooth in [0.0, 0.5, 1.0, 2.0]:
            self._test(
                lambda input_dim: T.DiscreteCosineTransform(smooth=smooth),
                autodiff=False,
            )

    def test_haar_transform(self):
        # NOTE: Need following since helper function unimplemented
        for flip in [True, False]:
            self._test(lambda input_dim: T.HaarTransform(flip=flip), autodiff=False)

    def test_elu(self):
        # NOTE: Need following since helper function mistakenly doesn't take input dim
        self._test(lambda input_dim: T.elu(), autodiff=False)

    def test_generalized_channel_permute(self):
        for shape in [(3, 16, 16), (1, 3, 32, 32), (2, 5, 3, 64, 64)]:
            # NOTE: Without changing the interface to generalized_channel_permute I can't reuse general
            # test for `event_dim > 1` transforms
            transform = T.generalized_channel_permute(channels=shape[-3])
            self._test_shape(shape, transform)
            self._test_inverse(shape, transform)

        for width_dim in [2, 4, 6]:
            input_dim = (width_dim**2) * 3
            self._test_jacobian(
                input_dim, Flatten(transform, (3, width_dim, width_dim))
            )

    def test_householder(self):
        self._test(partial(T.householder, count_transforms=2))

    def test_leaky_relu(self):
        # NOTE: Need following since helper function mistakenly doesn't take input dim
        self._test(lambda input_dim: T.leaky_relu(), autodiff=False)

    def test_lower_cholesky_affine(self):
        # NOTE: Need following since helper function unimplemented
        def transform_factory(input_dim):
            loc = torch.randn(input_dim)
            scale_tril = torch.randn(input_dim).exp().diag() + 0.03 * torch.randn(
                input_dim, input_dim
            )
            scale_tril = scale_tril.tril(0)
            return T.LowerCholeskyAffine(loc, scale_tril)

        self._test(transform_factory, autodiff=False)

    def test_matrix_exponential(self):
        self._test(T.matrix_exponential)

    def test_neural_autoregressive(self):
        for activation in ["ELU", "LeakyReLU", "sigmoid", "tanh"]:
            self._test(
                partial(T.neural_autoregressive, activation=activation), inverse=False
            )

    def test_ordered_transform(self):
        # NOTE: Need following since transform takes no input parameters
        self._test(lambda event_shape: T.OrderedTransform(), autodiff=False)

    def test_permute(self):
        for dim in [-1, -2]:
            self._test(partial(T.permute, dim=dim), event_dim=-dim, autodiff=False)

    def test_planar(self):
        self._test(T.planar, inverse=False)

    def test_polynomial(self):
        self._test(T.polynomial, inverse=False)

    def test_radial(self):
        self._test(T.radial, inverse=False)

    def test_simplex_to_ordered(self):
        self._test(
            lambda event_shape: T.SimplexToOrderedTransform(),
            shape=False,
            autodiff=False,
            base_dist_type="dirichlet",
        )
        # Unique shape behavior:
        assert T.SimplexToOrderedTransform().forward_shape((4, 3, 3)) == (4, 3, 2)
        assert T.SimplexToOrderedTransform().forward_shape((2,)) == (1,)
        assert T.SimplexToOrderedTransform().inverse_shape((2,)) == (3,)
        assert T.SimplexToOrderedTransform().inverse_shape((4, 3, 3)) == (4, 3, 4)

    def test_spline(self):
        for order in ["linear", "quadratic"]:
            self._test(partial(T.spline, order=order))

    def test_spline_coupling(self):
        self._test(T.spline_coupling)

    def test_spline_autoregressive(self):
        self._test(T.spline_autoregressive)

    def test_sylvester(self):
        self._test(T.sylvester, inverse=False)

    def test_normalize_transform(self):
        self._test(lambda p: T.Normalize(p=p))

    def test_softplus(self):
        self._test(lambda _: T.SoftplusTransform())

    def test_positive_power(self):
        for p in [0.3, 1.0, 3.0]:
            self._test(lambda _: T.PositivePowerTransform(p), event_dim=0)


@pytest.mark.parametrize("batch_shape", [(), (7,), (6, 5)])
@pytest.mark.parametrize("dim", [2, 3, 5])
@pytest.mark.parametrize(
    "transform",
    [
        T.CholeskyTransform(),
        T.CorrMatrixCholeskyTransform(),
    ],
    ids=lambda t: type(t).__name__,
)
def test_cholesky_transform(batch_shape, dim, transform):
    arange = torch.arange(dim)
    domain = transform.domain
    z = torch.randn(batch_shape + (dim * (dim - 1) // 2,))
    if domain is dist.constraints.corr_matrix:
        tril_mask = arange < arange.view(-1, 1)
    else:
        tril_mask = arange < arange.view(-1, 1) + 1
    x = transform.inv(T.CorrCholeskyTransform()(z))  # creates corr_matrix

    def vec_to_mat(x_vec):
        x_mat = x_vec.new_zeros(batch_shape + (dim, dim))
        x_mat[..., tril_mask] = x_vec
        x_mat = (
            x_mat
            + x_mat.transpose(-2, -1)
            - x_mat.diagonal(dim1=-2, dim2=-1).diag_embed()
        )
        if domain == dist.constraints.corr_matrix:
            x_mat = x_mat + x_mat.new_ones(x_mat.shape[-1]).diag_embed()
        return x_mat

    def transform_to_vec(x_vec):
        x_mat = vec_to_mat(x_vec)
        return transform(x_mat)[..., tril_mask]

    x_vec = x[..., tril_mask].clone().requires_grad_()
    x_mat = vec_to_mat(x_vec)
    y = transform(x_mat)
    log_det = transform.log_abs_det_jacobian(x_mat, y)
    if batch_shape == ():
        jacobian = torch.autograd.functional.jacobian(transform_to_vec, x_vec)
        assert_close(log_det, torch.slogdet(jacobian)[1])

    assert log_det.shape == batch_shape
    assert_close(y, torch.linalg.cholesky(x_mat))
    assert_close(transform.inv(y), x_mat)


@pytest.mark.parametrize("batch_shape", [(), (7,), (6, 5)])
@pytest.mark.parametrize("dim", [2, 3, 5])
@pytest.mark.parametrize(
    "transform",
    [
        T.LowerCholeskyTransform(),
        T.SoftplusLowerCholeskyTransform(),
        T.UnitLowerCholeskyTransform(),
    ],
    ids=lambda t: type(t).__name__,
)
def test_lower_cholesky_transform(transform, batch_shape, dim):
    shape = batch_shape + (dim, dim)
    x = torch.randn(shape)
    y = transform(x)
    assert y.shape == shape
    x2 = transform.inv(y)
    assert x2.shape == shape
    y2 = transform(x2)
    assert y2.shape == shape
    assert_close(y, y2)


@pytest.mark.parametrize("batch_shape", [(), (7,), (6, 7)])
@pytest.mark.parametrize("input_dim", [2, 3, 5])
@pytest.mark.parametrize("context_dim", [2, 3, 5])
def test_inverse_conditional_transform_module(batch_shape, input_dim, context_dim):
    cond_transform = T.conditional_spline(input_dim, context_dim, [6])

    noise = torch.rand(batch_shape + (input_dim,))
    context = torch.rand(batch_shape + (context_dim,))

    assert_close(
        cond_transform.inv.condition(context)(noise),
        cond_transform.condition(context).inv(noise),
    )

    assert cond_transform.inv.inv is cond_transform
    assert_close(
        cond_transform.inv.condition(context).inv(noise),
        cond_transform.condition(context).inv.inv(noise),
    )


@pytest.mark.parametrize("batch_shape", [(), (7,), (6, 7)])
@pytest.mark.parametrize("input_dim", [2, 3, 5])
@pytest.mark.parametrize("context_dim", [2, 3, 5])
@pytest.mark.parametrize("cache_size", [0, 1])
def test_conditional_compose_transform_module(
    batch_shape, input_dim, context_dim, cache_size
):
    conditional_transforms = [
        T.AffineTransform(1.0, 2.0),
        T.Spline(input_dim),
        T.conditional_spline(input_dim, context_dim, [5]),
        T.SoftplusTransform(),
        T.conditional_spline(input_dim, context_dim, [6]),
    ]
    cond_transform = dist.conditional.ConditionalComposeTransformModule(
        conditional_transforms, cache_size=cache_size
    )

    base_dist = dist.Normal(0, 1).expand(batch_shape + (input_dim,)).to_event(1)
    cond_dist = dist.ConditionalTransformedDistribution(base_dist, [cond_transform])

    context = torch.rand(batch_shape + (context_dim,))
    d = cond_dist.condition(context)
    transform = d.transforms[0]
    assert isinstance(transform, T.ComposeTransformModule)

    data = d.rsample()
    assert data.shape == batch_shape + (input_dim,)
    assert d.log_prob(data).shape == batch_shape

    actual_params = set(cond_transform.parameters())
    expected_params = set(
        torch.nn.ModuleList(
            [t for t in conditional_transforms if isinstance(t, torch.nn.Module)]
        ).parameters()
    )
    assert set() != actual_params == expected_params

    noise = base_dist.rsample()
    expected = noise
    for t in conditional_transforms:
        expected = (t.condition(context) if hasattr(t, "condition") else t)(expected)

    actual = transform(noise)
    assert_close(actual, expected)

    assert_close(cond_transform.inv.condition(context)(actual), noise)
    assert_close(cond_transform.condition(context).inv(expected), noise)
