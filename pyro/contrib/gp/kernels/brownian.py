from __future__ import absolute_import, division, print_function

from .kernel import Kernel


class Brownian(Kernel):
    """
    This kernel correponds to a two-sided Brownion motion (Wiener process):
    ``k(x, z) = min(|x|,|z|)`` if ``x.z >= 0`` and ``k(x, z) = 0`` otherwise.

    Note that the input dimension of this kernel must be 1.

    References:

    [1] `Theory and Statistical Applications of Stochastic Processes`,
    Yuliya Mishura, Georgiy Shevchenko
    """

    def __init__(self, input_dim, variance=None, active_dims=None, name="Brownian"):
        if input_dim != 1:
            raise ValueError("Input dimensional for Brownian kernel must be 1.")
        super(Brownian, self).__init__(input_dim, active_dims, name)

    def forward(self, X, Z=None, diag=False):
        variance = self.get_param("variance")
        if diag:
            return variance * X.abs().unsqueeze(1)

        if Z is None:
            Z = X
        return torch.where(X.sign() == Z.t().sign(),
                           torch.min(X.abs(), Z.t().abs()),
                           X.new(X.size(0), Z.size(0)).zero_())
