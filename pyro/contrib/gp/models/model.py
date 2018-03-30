from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.util import Parameterized
from pyro.infer import SVI
from pyro.optim import Adam, PyroOptim


class GPModel(Parameterized):
    """
    Base class for Gaussian Process (GP) models.

    GP models are :class:`Parameterized` subclasses. So its parameters can be learned,
    set priors, or fixed by using corresponding methods from :class:`Parameterized`.

    + You can use MCMC algorithms in :mod:`pyro.infer.mcmc` on :meth:`model` to get posterior samples for your test :func:`torch.tensor` 
      GP's parameters. For example:
    
        >>> xx
        >>> yy
        >>> zz
        
    + Using SVI on 
    
        >>> dd
        >>> dd
        >>> dd
    
    Or you can train with SVI using the pair :meth:`model`, :meth:`guide`
    as in `SVI tutorial <http://pyro.ai/examples/svi_part_i.html>`_. The source of
    :meth:`optimize` shows an example for how to use SVI for GP models.

    :param torch.Tensor X: A 1D or 2D input data for training. Its first dimension is
        the number of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        ``y.shape[-1]`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    :param str name: Name of this model.
    
    Example::

        >>> X = torch.tensor([[1, 5, 3], [4, 3, 7]])  # 2 data points
        >>> y = torch.tensor([2, 1])
        >>> kernel = gp.kernels.RBF(input_dim=3)
        >>> gpr = gp.models.GPRegression(X, y, kernel)
        >>> gpr.optimize()
        >>> Xnew = torch.tensor([[2, 3, 1]])  # 1 test data
        >>> f_loc, f_cov = gpr(Xnew, full_cov=True)

    """
    def __init__(self, X, y, kernel, jitter=1e-6, name=None):
        super(GPModel, self).__init__(name)
        self.set_data(X, y)
        self.kernel = kernel
        self.jitter = jitter

    def set_data(self, X, y=None):
        """
        Sets data for Gaussian Process models.

        :param torch.Tensor X: A 1D or 2D tensor of input data for training.
        :param torch.Tensor y: A tensor of output data for training with
            ``y.shape[-1]`` equals to number of data points.
        """
        if X.dim() > 2:
            raise ValueError("Expected input tensor of 1 or 2 dimensions, "
                             "actual dim = {}".format(X.dim()))
        if y is not None and X.shape[0] != y.shape[-1]:
            raise ValueError("Expected the number of data inputs equal to the number "
                             "of data outputs, but got {} and {}."
                             .format(X.shape[0], y.shape[-1]))
        self.X = X
        self.y = y

    def model(self):
        """
        A "model" stochastic method.
        """
        raise NotImplementedError

    def guide(self):
        """
        A "guide" stochastic method.
        """
        raise NotImplementedError

    def optimize(self, optimizer=Adam({}), num_steps=1000):
        """
        A convenient method to optimize parameters for the Gaussian Process model
        using SVI.

        :param PyroOptim optimizer: Optimizer.
        :param int num_steps: Number of steps to run SVI.
        :returns: losses of the training procedure
        :rtype: list
        """
        if not isinstance(optimizer, PyroOptim):
            raise ValueError("Optimizer should be an instance of "
                             "pyro.optim.PyroOptim class.")
        svi = SVI(self.model, self.guide, optimizer, loss="ELBO")
        losses = []
        for i in range(num_steps):
            losses.append(svi.step())
        return losses

    def forward(self, *args, **kwargs):
        """
        Implements prediction step.
        """
        raise NotImplementedError

    def _check_Xnew_shape(self, Xnew):
        """
        Checks the correction of the shape of new data.
        """
        if Xnew.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same number of dimensions.")
        if Xnew.dim() == 2 and self.X.shape[1] != Xnew.shape[1]:
            raise ValueError("Train data and test data should have the same feature sizes.")
