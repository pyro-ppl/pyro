# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import Transform, constraints

from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.util import copy_docs_from
from pyro.nn import DenseNN


@copy_docs_from(Transform)
class ConditionedGeneralizedChannelPermute(Transform):
    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 3

    def __init__(self, permutation=None, LU=None):
        super(ConditionedGeneralizedChannelPermute, self).__init__(cache_size=1)

        self.permutation = permutation
        self.LU = LU

    @property
    def U_diag(self):
        return self.LU.diag()

    @property
    def L(self):
        return self.LU.tril(diagonal=-1) + torch.eye(self.LU.size(-1), dtype=self.LU.dtype, device=self.LU.device)

    @property
    def U(self):
        return self.LU.triu()

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        """
        NOTE: As is the case for other conditional transforms, the batch dim of the
        context variable (reflected in the initial dimensions of filters in this
        case), if this is a conditional transform, must broadcast over the batch dim
        of the input variable.

        Also, the reason the following line uses matrix multiplication rather than
        F.conv2d is so we can perform multiple convolutions when the filters
        "kernel" has batch dimensions
        """
        filters = (self.permutation @ self.L @ self.U)[..., None, None]
        y = (filters * x.unsqueeze(-4)).sum(-3)
        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """

        """
        NOTE: This method is equivalent to the following two lines. Using
        Tensor.inverse() would be numerically unstable, however.

        filters = (self.permutation @ self.L @ self.U).inverse()[..., None, None]
        x = F.conv2d(y.view(-1, *y.shape[-3:]), filters)
        return x.view_as(y)

        """

        # Do a matrix vector product over the channel dimension
        # in order to apply inverse permutation matrix
        y_flat = y.flatten(start_dim=-2)
        LUx = (y_flat.unsqueeze(-3) * self.permutation.T.unsqueeze(-1)).sum(-2)

        # Solve L(Ux) = P^1y
        Ux, _ = torch.triangular_solve(LUx, self.L, upper=False)

        # Solve Ux = (PL)^-1y
        x, _ = torch.triangular_solve(Ux, self.U)

        # Unflatten x (works when context variable has batch dim)
        return x.reshape(x.shape[:-1] + y.shape[-2:])

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(det(dy/dx))).
        """

        h, w = x.shape[-2:]
        log_det = h * w * self.U_diag.abs().log().sum()
        return log_det * torch.ones(x.size()[:-3], dtype=x.dtype, layout=x.layout, device=x.device)


@copy_docs_from(ConditionedGeneralizedChannelPermute)
class GeneralizedChannelPermute(ConditionedGeneralizedChannelPermute, TransformModule):
    r"""
    A bijection that generalizes a permutation on the channels of a batch of 2D
    image in :math:`[\ldots,C,H,W]` format. Specifically this transform performs
    the operation,

        :math:`\mathbf{y} = \text{torch.nn.functional.conv2d}(\mathbf{x}, W)`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    and :math:`W\sim C\times C\times 1\times 1` is the filter matrix for a 1x1
    convolution with :math:`C` input and output channels.

    Ignoring the final two dimensions, :math:`W` is restricted to be the matrix
    product,

        :math:`W = PLU`

    where :math:`P\sim C\times C` is a permutation matrix on the channel
    dimensions, :math:`L\sim C\times C` is a lower triangular matrix with ones on
    the diagonal, and :math:`U\sim C\times C` is an upper triangular matrix.
    :math:`W` is initialized to a random orthogonal matrix. Then, :math:`P` is fixed
    and the learnable parameters set to :math:`L,U`.

    The input :math:`\mathbf{x}` and output :math:`\mathbf{y}` both have shape
    `[...,C,H,W]`, where `C` is the number of channels set at initialization.

    This operation was introduced in [1] for Glow normalizing flow, and is also
    known as 1x1 invertible convolution. It appears in other notable work such as
    [2,3], and corresponds to the class `tfp.bijectors.MatvecLU` of TensorFlow
    Probability.

    Example usage:

    >>> channels = 3
    >>> base_dist = dist.Normal(torch.zeros(channels, 32, 32),
    ... torch.ones(channels, 32, 32))
    >>> inv_conv = GeneralizedChannelPermute(channels=channels)
    >>> flow_dist = dist.TransformedDistribution(base_dist, [inv_conv])
    >>> flow_dist.sample()  # doctest: +SKIP

    :param channels: Number of channel dimensions in the input.
    :type channels: int

    [1] Diederik P. Kingma, Prafulla Dhariwal. Glow: Generative Flow with Invertible
    1x1 Convolutions. [arXiv:1807.03039]

    [2] Ryan Prenger, Rafael Valle, Bryan Catanzaro. WaveGlow: A Flow-based
    Generative Network for Speech Synthesis. [arXiv:1811.00002]

    [3] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural Spline
    Flows. [arXiv:1906.04032]

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 3

    def __init__(self, channels=3, permutation=None):
        super(GeneralizedChannelPermute, self).__init__()
        self.__delattr__('permutation')

        # Sample a random orthogonal matrix
        W, _ = torch.qr(torch.randn(channels, channels))

        # Construct the partially pivoted LU-form and the pivots
        LU, pivots = W.lu()

        # Convert the pivots into the permutation matrix
        if permutation is None:
            P, _, _ = torch.lu_unpack(LU, pivots)
        else:
            if len(permutation) != channels:
                raise ValueError(
                    'Keyword argument "permutation" expected to have {} elements but {} found.'.format(
                        channels, len(permutation)))
            P = torch.eye(channels, channels)[permutation.type(dtype=torch.int64)]

        # We register the permutation matrix so that the model can be serialized
        self.register_buffer('permutation', P)

        # NOTE: For this implementation I have chosen to store the parameters densely, rather than
        # storing L, U, and s separately
        self.LU = torch.nn.Parameter(LU)


@copy_docs_from(ConditionalTransformModule)
class ConditionalGeneralizedChannelPermute(ConditionalTransformModule):
    r"""
    A bijection that generalizes a permutation on the channels of a batch of 2D
    image in :math:`[\ldots,C,H,W]` format conditioning on an additional context
    variable. Specifically this transform performs the operation,

        :math:`\mathbf{y} = \text{torch.nn.functional.conv2d}(\mathbf{x}, W)`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    and :math:`W\sim C\times C\times 1\times 1` is the filter matrix for a 1x1
    convolution with :math:`C` input and output channels.

    Ignoring the final two dimensions, :math:`W` is restricted to be the matrix
    product,

        :math:`W = PLU`

    where :math:`P\sim C\times C` is a permutation matrix on the channel
    dimensions, and  :math:`LU\sim C\times C` is an invertible product of a lower
    triangular and an upper triangular matrix that is the output of an NN with
    input :math:`z\in\mathbb{R}^{M}` representing the context variable to
    condition on.

    The input :math:`\mathbf{x}` and output :math:`\mathbf{y}` both have shape
    `[...,C,H,W]`, where `C` is the number of channels set at initialization.

    This operation was introduced in [1] for Glow normalizing flow, and is also
    known as 1x1 invertible convolution. It appears in other notable work such as
    [2,3], and corresponds to the class `tfp.bijectors.MatvecLU` of TensorFlow
    Probability.

    Example usage:

    >>> from pyro.nn.dense_nn import DenseNN
    >>> context_dim = 5
    >>> batch_size = 3
    >>> channels = 3
    >>> base_dist = dist.Normal(torch.zeros(channels, 32, 32),
    ... torch.ones(channels, 32, 32))
    >>> hidden_dims = [context_dim*10, context_dim*10]
    >>> nn = DenseNN(context_dim, hidden_dims, param_dims=[channels*channels])
    >>> transform = ConditionalGeneralizedChannelPermute(nn, channels=channels)
    >>> z = torch.rand(batch_size, context_dim)
    >>> flow_dist = dist.ConditionalTransformedDistribution(base_dist,
    ... [transform]).condition(z)
    >>> flow_dist.sample(sample_shape=torch.Size([batch_size])) # doctest: +SKIP

    :param nn: a function inputting the context variable and outputting
        real-valued parameters of dimension :math:`C^2`.
    :param channels: Number of channel dimensions in the input.
    :type channels: int

    [1] Diederik P. Kingma, Prafulla Dhariwal. Glow: Generative Flow with Invertible
    1x1 Convolutions. [arXiv:1807.03039]

    [2] Ryan Prenger, Rafael Valle, Bryan Catanzaro. WaveGlow: A Flow-based
    Generative Network for Speech Synthesis. [arXiv:1811.00002]

    [3] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural Spline
    Flows. [arXiv:1906.04032]

    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    event_dim = 3

    def __init__(self, nn, channels=3, permutation=None):
        super().__init__()
        self.nn = nn
        self.channels = channels
        if permutation is None:
            permutation = torch.randperm(channels, device='cpu').to(torch.Tensor().device)
        P = torch.eye(len(permutation), len(permutation))[permutation.type(dtype=torch.int64)]
        self.register_buffer('permutation', P)

    def condition(self, context):
        LU = self.nn(context)
        LU = LU.view(LU.shape[:-1] + (self.channels, self.channels))
        return ConditionedGeneralizedChannelPermute(self.permutation, LU)


def generalized_channel_permute(**kwargs):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.GeneralizedChannelPermute` object for
    consistency with other helpers.

    :param channels: Number of channel dimensions in the input.
    :type channels: int

    """

    return GeneralizedChannelPermute(**kwargs)


def conditional_generalized_channel_permute(context_dim, channels=3, hidden_dims=None):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.ConditionalGeneralizedChannelPermute`
    object for consistency with other helpers.

    :param channels: Number of channel dimensions in the input.
    :type channels: int

    """

    if hidden_dims is None:
        hidden_dims = [channels * 10, channels * 10]
    nn = DenseNN(context_dim, hidden_dims, param_dims=[channels * channels])
    return ConditionalGeneralizedChannelPermute(nn, channels)
