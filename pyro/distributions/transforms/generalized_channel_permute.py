import torch
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from
from pyro.distributions.torch_transform import TransformModule


@copy_docs_from(TransformModule)
class GeneralizedChannelPermute(TransformModule):
    """
    A bijection that generalizes a permutation on the channels of a batch of 2D
    image in :math:`[\\ldots,C,H,W]` format. Specifically this transform performs
    the operation,

        :math:`\\mathbf{y} = \\text{torch.nn.functional.conv2d}(\\mathbf{x}, W)`

    where :math:`\\mathbf{x}` are the inputs, :math:`\\mathbf{y}` are the outputs,
    and :math:`W\\sim C\\times C\\times 1\\times 1` is the filter matrix for a 1x1
    convolution with :math:`C` input and output channels.

    Ignoring the final two dimensions, :math:`W` is restricted to be the matrix
    product,

        :math:`W = PLU`

    where :math:`P\\sim C\\times C` is a permutation matrix on the channel
    dimensions, :math:`L\\sim C\\times C` is a lower triangular matrix with ones on
    the diagonal, and :math:`U\\sim C\\times C` is an upper triangular matrix.
    :math:`W` is initialized to a random orthogonal matrix. Then, :math:`P` is fixed
    and the learnable parameters set to :math:`L,U`.

    The input :math:`\\mathbf{x}` and output :math:`\\mathbf{y}` both have shape
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

    def __init__(self, channels=3):
        super(GeneralizedChannelPermute, self).__init__(cache_size=1)
        self.channels = channels

        # Sample a random orthogonal matrix
        W, _ = torch.qr(torch.randn(channels, channels))

        # Construct the partially pivoted LU-form and the pivots
        LU, pivots = W.lu()

        # Convert the pivots into the permutation matrix
        P, _, _ = torch.lu_unpack(LU, pivots)

        # We register the permutation matrix so that the model can be serialized
        self.register_buffer('permutation', P)

        # NOTE: For this implementation I have chosen to store the parameters densely, rather than
        # storing L, U, and s separately
        self.LU = torch.nn.Parameter(LU)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """

        # Extract the lower and upper matrices from the packed LU matrix
        U = self.LU.triu()
        L = self.LU.tril()
        L.diagonal(dim1=-2, dim2=-1).fill_(1)

        # Perform the 2D convolution, using the weight
        filters = (self.permutation @ L @ U)[..., None, None]
        y = F.conv2d(x.view(-1, *x.shape[-3:]), filters)

        return y.view_as(x)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """

        """
        NOTE: This method is equivalent to the following two lines. Using
        Tensor.inverse() would be numerically unstable, however.

        U = self.LU.triu()
        L = self.LU.tril()
        L.diagonal(dim1=-2, dim2=-1).fill_(1)
        filters = (self.permutation @ L @ U).inverse()[..., None, None]
        x = F.conv2d(y.view(-1, *y.shape[-3:]), filters)
        return x.view_as(y)

        """

        # Do a matrix vector product over the channel dimension
        # in order to apply inverse permutation matrix
        y_flat = y.flatten(start_dim=-2)
        LUx = (y_flat.unsqueeze(-3) * self.permutation.T.unsqueeze(-1)).sum(-2)

        # Solve L(Ux) = P^1y
        U = torch.triu(self.LU)
        L = self.LU.tril(-1) + torch.eye(self.LU.size(-1), dtype=self.LU.dtype, device=self.LU.device)
        Ux, _ = torch.triangular_solve(LUx, L, upper=False)

        # Solve Ux = (PL)^-1y
        x, _ = torch.triangular_solve(Ux, U)

        # Unflatten x
        return x.view_as(y)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(det(dy/dx))).
        """

        h, w = x.shape[-2:]
        log_det = h * w * self.LU.diag().abs().log().sum()

        return log_det * torch.ones(x.size()[:-3], dtype=x.dtype, layout=x.layout, device=x.device)


def generalized_channel_permute(**kwargs):
    """
    A helper function to create a
    :class:`~pyro.distributions.transforms.GeneralizedChannelPermute` object for
    consistency with other helpers.

    :param channels: Number of channel dimensions in the input.
    :type channels: int

    """

    return GeneralizedChannelPermute(**kwargs)
