from __future__ import absolute_import, division, print_function

from pyro.contrib.gp.util import Parameterized
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, PyroOptim


def _zero_mean_function(x):
    return 0


class GPModel(Parameterized):
    """
    Base class for Gaussian Process models.

    The core of a Gaussian Process is a covariance function :math:`k` which governs
    the similarity between input points. Given :math:`k`, we can establish a
    distribution over functions :math:`f` by a multivarite normal distribution

    .. math:: p(f(X)) = \mathcal{N}(0, k(X, X)),

    where :math:`X` is any set of input points and :math:`k(X, X)` is a covariance
    matrix whose entries are outputs :math:`k(x, z)` of :math:`k` over input pairs
    :math:`(x, z)`. This distribution is usually denoted by

    .. math:: f \sim \mathcal{GP}(0, k).

    .. note:: Generally, beside a covariance matrix :math:`k`, a Gaussian Process can
        also be specified by a mean function :math:`m` (which is a zero-value function
        by default). In that case, its distribution will be

        .. math:: p(f(X)) = \mathcal{N}(m(X), k(X, X)).

    Gaussian Process models are :class:`~pyro.contrib.gp.util.Parameterized`
    subclasses. So its parameters can be learned, set priors, or fixed by using
    corresponding methods from :class:`~pyro.contrib.gp.util.Parameterized`. A typical
    way to define a Gaussian Process model is

        >>> X = torch.tensor([[1., 5, 3], [4, 3, 7]])
        >>> y = torch.tensor([2., 1])
        >>> kernel = gp.kernels.RBF(input_dim=3)
        >>> kernel.set_prior("variance", dist.Uniform(torch.tensor(0.5), torch.tensor(1.5)))
        >>> kernel.set_prior("lengthscale", dist.Uniform(torch.tensor(1.0), torch.tensor(3.0)))
        >>> gpr = gp.models.GPRegression(X, y, kernel)

    There are two ways to train a Gaussian Process model:

    + Using an MCMC algorithm (in module :mod:`pyro.infer.mcmc`) on :meth:`model` to
      get posterior samples for the Gaussian Process's parameters. For example:

        >>> hmc_kernel = HMC(gpr.model)
        >>> mcmc_run = MCMC(hmc_kernel, num_samples=10)
        >>> posterior_ls_trace = []  # store lengthscale trace
        >>> ls_name = param_with_module_name(gpr.kernel.name, "lengthscale")
        >>> for trace, _ in mcmc_run._traces():
        ...     posterior_ls_trace.append(trace.nodes[ls_name]["value"])

    + Using a variational inference (e.g. :class:`~pyro.infer.svi.SVI`) on the pair
      :meth:`model`, :meth:`guide` as in `SVI tutorial
      <http://pyro.ai/examples/svi_part_i.html>`_:

        >>> optimizer = pyro.optim.Adam({"lr": 0.01})
        >>> svi = SVI(gpr.model, gpr.guide, optimizer, loss=Trace_ELBO())
        >>> for i in range(1000):
        ...     svi.step()  # doctest: +SKIP

    To give a prediction on new dataset, simply use :meth:`forward` like any PyTorch
    :class:`torch.nn.Module`:

        >>> Xnew = torch.tensor([[2., 3, 1]])
        >>> f_loc, f_cov = gpr(Xnew, full_cov=True)

    Reference:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor X: A input data for training. Its first dimension is the number
        of data points.
    :param torch.Tensor y: An output data for training. Its last dimension is the
        number of data points.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object, which
        is the covariance function :math:`k`.
    :param callable mean_function: An optional mean function :math:`m` of this Gaussian
        process. By default, we use zero mean.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :param str name: Name of this model.
    """
    def __init__(self, X, y, kernel, mean_function=None, jitter=1e-6, name=None):
        super(GPModel, self).__init__(name)
        self.set_data(X, y)
        self.kernel = kernel
        self.mean_function = (mean_function if mean_function is not None else
                              _zero_mean_function)
        self.jitter = jitter

    def model(self):
        """
        A "model" stochastic function. If ``self.y`` is ``None``, this method returns
        mean and variance of the Gaussian Process prior.
        """
        raise NotImplementedError

    def guide(self):
        """
        A "guide" stochastic function to be used in variational inference methods. It
        also gives posterior information to the method :meth:`forward` for prediction.
        """
        raise NotImplementedError

    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, \theta),

        where :math:`\theta` are parameters of this model.

        .. note:: Model's parameters :math:`\theta` together with kernel's parameters
            have been learned from a training procedure (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        raise NotImplementedError

    def set_data(self, X, y=None):
        """
        Sets data for Gaussian Process models.

        Some examples to utilize this method are:

        .. doctest::
           :hide:

            >>> X = torch.tensor([[1., 5, 3], [4, 3, 7]])
            >>> y = torch.tensor([2., 1])
            >>> kernel = gp.kernels.RBF(input_dim=3)
            >>> kernel.set_prior("variance", dist.Uniform(torch.tensor(0.5), torch.tensor(1.5)))
            >>> kernel.set_prior("lengthscale", dist.Uniform(torch.tensor(1.0), torch.tensor(3.0)))
            >>> optimizer = pyro.optim.Adam({"lr": 0.01})

        + Batch training on a sparse variational model:

            >>> Xu = torch.tensor([[1., 0, 2]])  # inducing input
            >>> likelihood = gp.likelihoods.Gaussian()
            >>> vsgp = gp.models.VariationalSparseGP(X, y, kernel, Xu, likelihood)
            >>> svi = SVI(vsgp.model, vsgp.guide, optimizer, Trace_ELBO())
            >>> batched_X, batched_y = X.split(split_size=10), y.split(split_size=10)
            >>> for Xi, yi in zip(batched_X, batched_y):
            ...     vsgp.set_data(Xi, yi)
            ...     svi.step()  # doctest: +SKIP

        + Making a two-layer Gaussian Process stochastic function:


            >>> gpr1 = gp.models.GPRegression(X, None, kernel, name="GPR1")
            >>> Z, _ = gpr1.model()
            >>> gpr2 = gp.models.GPRegression(Z, y, kernel, name="GPR2")
            >>> def two_layer_model():
            ...     Z, _ = gpr1.model()
            ...     gpr2.set_data(Z, y)
            ...     return gpr2.model()

        References:

        [1] `Scalable Variational Gaussian Process Classification`,
        James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani

        [2] `Deep Gaussian Processes`,
        Andreas C. Damianou, Neil D. Lawrence

        :param torch.Tensor X: A input data for training. Its first dimension is the
            number of data points.
        :param torch.Tensor y: An output data for training. Its last dimension is the
            number of data points.
        """
        if y is not None and X.shape[0] != y.shape[-1]:
            raise ValueError("Expected the number of input data points equal to the "
                             "number of output data points, but got {} and {}."
                             .format(X.shape[0], y.shape[-1]))
        self.X = X
        self.y = y

    def optimize(self, optimizer=None, loss=None, num_steps=1000):
        """
        A convenient method to optimize parameters for the Gaussian Process model
        using :class:`~pyro.infer.svi.SVI`.

        :param PyroOptim optimizer: A Pyro optimizer.
        :param ELBO loss: A Pyro loss instance.
        :param int num_steps: Number of steps to run SVI.
        :returns: a list of losses during the training procedure
        :rtype: list
        """
        if optimizer is None:
            optimizer = Adam({})
        if not isinstance(optimizer, PyroOptim):
            raise ValueError("Optimizer should be an instance of "
                             "pyro.optim.PyroOptim class.")
        if loss is None:
            loss = Trace_ELBO()
        svi = SVI(self.model, self.guide, optimizer, loss=loss)
        losses = []
        for i in range(num_steps):
            losses.append(svi.step())
        return losses

    def _check_Xnew_shape(self, Xnew):
        """
        Checks the correction of the shape of new data.

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        """
        if Xnew.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same "
                             "number of dimensions, but got {} and {}."
                             .format(self.X.dim(), Xnew.dim()))
        if self.X.shape[1:] != Xnew.shape[1:]:
            raise ValueError("Train data and test data should have the same "
                             "shape of features, but got {} and {}."
                             .format(self.X.shape[1:], Xnew.shape[1:]))
