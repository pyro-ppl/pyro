from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
from pyro.contrib.gp.util import Parameterized
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from pyro.params import param_with_module_name


class GPLVM(Parameterized):
    """
    Gaussian Process Latent Variable Model (GPLVM) model.

    GPLVM is a Gaussian Process model with its train input data is a latent variable.
    This model is useful for dimensional reduction of high dimensional data. Assume the
    mapping from low dimensional latent variable to is a Gaussian Process instance.
    Then the high dimensional data will play the role of train output ``y`` and our
    target is to learn latent inputs which best explain ``y``. For the purpose of
    dimensional reduction, latent inputs should have lower dimensions than ``y``.

    We follows reference [1] to put a unit Gaussian prior to the input and approximate
    its posterior by a multivariate normal distribution with two variational
    parameters: ``X_loc`` and ``X_scale_tril``.

    For example, we can do dimensional reduction on Iris dataset as follows:

        >>> # With y as the 2D Iris data of shape 150x4 and we want to reduce its dimension
        >>> # to a tensor X of shape 150x2, we will use GPLVM.

        .. doctest::
           :hide:

            >>> # Simulating iris data.
            >>> y = torch.stack([dist.Normal(4.8, 0.1).sample((150,)),
            ...                 dist.Normal(3.2, 0.3).sample((150,)),
            ...                 dist.Normal(1.5, 0.4).sample((150,)),
            ...                 dist.Exponential(0.5).sample((150,))])

        >>> # First, define the initial values for X parameter:
        >>> X_init = torch.zeros(150, 2)
        >>> # Then, define a Gaussian Process model with input X and output y:
        >>> kernel = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))
        >>> Xu = torch.zeros(20, 2)  # initial inducing inputs of sparse model
        >>> gpmodel = gp.models.SparseGPRegression(X_init, y, kernel, Xu)
        >>> # Finally, wrap gpmodel by GPLVM, optimize, and get the "learned" mean of X:
        >>> gplvm = gp.models.GPLVM(gpmodel)
        >>> gplvm.optimize()  # doctest: +SKIP
        >>> print(gplvm.X)

    Reference:

    [1] Bayesian Gaussian Process Latent Variable Model
    Michalis K. Titsias, Neil D. Lawrence

    :param ~pyro.contrib.gp.models.model.GPModel base_model: A Pyro Gaussian Process
        model object. Note that ``base_model.X`` will be the initial value for the
        variational parameter ``X_loc``.
    :param str name: Name of this model.
    """
    def __init__(self, base_model, name="GPLVM"):
        super(GPLVM, self).__init__(name)
        if base_model.X.dim() != 2:
            raise ValueError("GPLVM model only works with 2D latent X, but got "
                             "X.dim() = {}.".format(base_model.X.dim()))
        self.base_model = base_model
        self.X = Parameter(self.base_model.X)
        self.set_prior("X", )
        self.set_guide("X", dist.MultivariateNormal)
        self._call_base_model_guide = True

    def model(self):
        self.mode = "model"
        self.base_model.set_data(self.X, self.y)
        self.base_model.model()

    def guide(self):
        self.mode = "guide"
        self.base_model.set_data(self.X, self.y)
        self.base_model.guide()

    def forward(self, **kwargs):
        """
        Forward method has the same signal as its ``base_model``.
        """
        self.mode = "guide"
        self.base_model.set_data(self.X, self.y)
        return self.base_model(**kwargs)

    def optimize(self, optimizer=None, loss=None, num_steps=1000):
        """
        A convenient method to optimize parameters for GPLVM model using
        :class:`~pyro.infer.svi.SVI`.

        :param ~optim.PyroOptim optimizer: A Pyro optimizer.
        :param ELBO loss: A Pyro loss instance.
        :param int num_steps: Number of steps to run SVI.
        :returns: a list of losses during the training procedure
        :rtype: list
        """
        optimizer = Adam({"lr": 0.01}) if optimizer is None else optimizer
        if not isinstance(optimizer, optim.PyroOptim):
            raise ValueError("Optimizer should be an instance of "
                             "pyro.optim.PyroOptim class.")
        loss = Trace_ELBO() if loss is not None else loss
        svi = SVI(self.model, self.guide, optimizer, loss=loss)
        losses = []
        for i in range(num_steps):
            losses.append(svi.step())
        return losses
