from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
from torch.nn import Parameter

import pyro
from pyro.contrib.gp.util import Parameterized
import pyro.distributions as dist
import pyro.infer as infer
import pyro.optim as optim


class GPLVM(Parameterized):
    """
    Gaussian Process Latent Variable Model (GPLVM) model.

    GPLVM is a Gaussian Process model with its train input data is a latent variable.
    This model is useful for dimensional reduction of high dimensional data. Assume
    the mapping from low dimensional latent variable to is a Gaussian Process
    instance. Then the high dimensional data will play the role of train output ``y``
    and our target is to learn latent inputs which best explain ``y``. For the purpose
    of dimensional reduction, latent inputs should have lower dimensions than ``y``.

    We follows reference [1] to put a unit Gaussian prior to the input and approximate
    its posterior by a multivariate normal distribution with two variational
    parameters: ``X_loc`` and ``X_scale_tril``.

    Reference:

    [1] Bayesian Gaussian Process Latent Variable Model
    Michalis K. Titsias, Neil D. Lawrence

    :param pyro.contrib.gp.models.model.GPModel base_model: A Pyro Gaussian Process
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
        self.y = self.base_model.y

        self.X_loc = Parameter(self.base_model.X)

        C = self.X_loc.shape[1]
        X_scale_tril_shape = self.X_loc.shape + (C,)
        Id = torch.eye(C, out=self.X_loc.new_empty(C, C))
        X_scale_tril = Id.expand(X_scale_tril_shape)
        self.X_scale_tril = Parameter(X_scale_tril)
        self.set_constraint("X_scale_tril", constraints.lower_cholesky)

        self._call_base_model_guide = True

    def model(self):
        self.set_mode("model", only_this_module=True)

        zero_loc = self.X_loc.new_zeros(self.X_loc.shape)
        C = self.X_loc.shape[1]
        Id = torch.eye(C, out=self.X_loc.new_empty(C, C))
        X_name = pyro.param_with_module_name(self.name, "X")
        X = pyro.sample(X_name, dist.MultivariateNormal(zero_loc, scale_tril=Id)
                                    .independent(zero_loc.dim()-1))

        self.base_model.set_data(X, self.y)
        self.base_model.model()

    def guide(self):
        self.set_mode("guide", only_this_module=True)

        X_loc = self.get_param("X_loc")
        X_scale_tril = self.get_param("X_scale_tril")
        X_name = pyro.param_with_module_name(self.name, "X")
        X = pyro.sample(X_name,
                        dist.MultivariateNormal(X_loc, scale_tril=X_scale_tril)
                            .independent(X_loc.dim()-1))

        self.base_model.set_data(X, self.y)
        if self._call_base_model_guide:
            self.base_model.guide()

    def forward(self, **kwargs):
        """
        Forward method has the same signal as its ``base_model``. Note that the train
        input data of ``base_model`` is sampled from GPLVM.
        """
        # avoid calling base_model's guide two times
        self._call_base_model_guide = False
        self.guide()
        self._call_base_model_guide = True
        return self.base_model(**kwargs)

    def optimize(self, optimizer=optim.Adam({}), num_steps=1000):
        """
        A convenient method to optimize parameters for GPLVM model using
        :class:`~pyro.infer.svi.SVI`.

        :param PyroOptim optimizer: A Pyro optimizer.
        :param int num_steps: Number of steps to run SVI.
        :returns: a list of losses during the training procedure
        :rtype: list
        """
        if not isinstance(optimizer, optim.PyroOptim):
            raise ValueError("Optimizer should be an instance of "
                             "pyro.optim.PyroOptim class.")
        svi = infer.SVI(self.model, self.guide, optimizer, loss=infer.Trace_ELBO())
        losses = []
        for i in range(num_steps):
            losses.append(svi.step())
        return losses
