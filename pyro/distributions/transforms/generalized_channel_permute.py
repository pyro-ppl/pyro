import torch
from torch.distributions.utils import lazy_property
from torch.distributions import constraints
import torch.nn.functional as F

from pyro.distributions.util import copy_docs_from
from pyro.distributions.torch_transform import TransformModule

@copy_docs_from(TransformModule)
class GeneralizedChannelPermute(TransformModule):
    """
    A bijection that rotates each pixel of a 2D image in 3D channel space. That is, a single 3x3 rotation matrix is
    applied to each [3,1,1] dimensional pixel. This operation was introduced in  is also known as 1x1


    This is useful in between :class:`~pyro.distributions.transforms.AffineAutoregressive` transforms to increase the
    flexibility of the resulting distribution and stabilize learning. Whilst not being an autoregressive transform,
    the log absolute determinate of the Jacobian is easily calculable as 0. Note that reordering the input dimension
    between two layers of :class:`~pyro.distributions.transforms.AffineAutoregressive` is not equivalent to reordering
    the dimension inside the MADE networks that those IAFs use; using a :class:`~pyro.distributions.transforms.Permute`
    transform results in a distribution with more flexibility.

    Example usage:

    >>> from pyro.nn import AutoRegressiveNN
    >>> from pyro.distributions.transforms import AffineAutoregressive, Permute
    >>> base_dist = dist.Normal(torch.zeros(10), torch.ones(10))
    >>> iaf1 = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> ff = Permute(torch.randperm(10, dtype=torch.long))
    >>> iaf2 = AffineAutoregressive(AutoRegressiveNN(10, [40]))
    >>> flow_dist = dist.TransformedDistribution(base_dist, [iaf1, ff, iaf2])
    >>> flow_dist.sample()  # doctest: +SKIP
        tensor([-0.4071, -0.5030,  0.7924, -0.2366, -0.2387, -0.1417,  0.0868,
                0.1389, -0.4629,  0.0986])

    :param permutation: a permutation ordering that is applied to the inputs.
    :type permutation: torch.LongTensor

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
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """

        # Extract the lower and upper matrices from the packed LU matrix
        U = self.LU.triu()
        L = self.LU.tril()
        L.diagonal(dim1=-2, dim2=-1).fill_(1)

        # NOTE: Would this be a better way to extract L? Would need to apply device/type of torch.ones
        #L = torch.tril(self.LU, diagonal = -1) + torch.diag(torch.ones(self.channels))

        # Perform the 2D convolution, using the weight
        filters = (self.permutation @ L @ U)[..., None, None]
        y = F.conv2d(x, filters)

        return y

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x.
        """

        # NOTE: The following lines are equivalent to these two.
        # Using Tensor.inverse() would be the numerically unstable, however.
        #filters = (self.permutation @ L @ U).inverse()[..., None, None]
        #x = F.conv2d(y, filters)

        # Do a matrix vector product over the channel dimension
        # in order to apply inverse permutation matrix
        y_flat = y.flatten(start_dim=-2)
        LUx = (y_flat.unsqueeze(-3) * self.permutation.T.unsqueeze(-1)).sum(-3)

        # Solve L(Ux) = P^1y
        U = torch.triu(self.LU)
        L = self.LU.tril()
        L.diagonal(dim1=-2, dim2=-1).fill_(1)
        Ux, _ = torch.triangular_solve(LUx, L, upper=False)

        # Solve Ux = (PL)^-1y
        x, _ = torch.triangular_solve(Ux, U)
    
        # Unflatten x
        return x.view_as(y)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e. log(abs([dy_0/dx_0, ..., dy_{N-1}/dx_{N-1}])).
        Note that this type of transform is not autoregressive, so the log Jacobian is not the sum of the previous
        expression. However, it turns out it's always 0 (since the determinant is -1 or +1), and so returning a
        vector of zeros works.
        """

        h, w = x.shape[-2:]
        log_det = h * w * torch.log(torch.abs(torch.diag(self.LU))).sum()

        return log_det*torch.ones(x.size()[:-3], dtype=x.dtype, layout=x.layout, device=x.device)


def generalized_channel_permute(input_dim, **kwargs):
    """
    A helper function to create a :class:`~pyro.distributions.transforms.Permute` object for consistency with other
    helpers.

    :param input_dim: Dimension of input variable
    :type input_dim: int
    :param permutation: Torch tensor of integer indices representing permutation. Defaults
        to a random permutation.
    :type permutation: torch.LongTensor

    """

    return GeneralizedChannelPermute(**kwargs)
