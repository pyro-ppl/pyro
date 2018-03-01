from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import numbers

from pyro.contrib.gp import Parameterized


class Kernel(Parameterized):
    """
    Base class for kernels used in Gaussian Process.

    Every inherited class should implement the forward pass which
    take inputs X, Z and return their covariance matrix.

    To construct a new kernel from the old ones, we can use Python operators
    `+` or `*` with another kernel/constant. Or we can use methods `.add(...)`,
    `.mul(...)`, `.exp(...)`, `.warp(...)`, `.vertical_scale(...)`. See the
    documentation of these methods for more information.

    References:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param int input_dim: Dimension of inputs for this kernel.
    :param list active_dims: List of dimensions of the input which the kernel acts on.
    :param str name: Name of the kernel.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        super(Kernel, self).__init__(name)
        if active_dims is None:
            active_dims = slice(input_dim)
        elif input_dim != len(active_dims):
            raise ValueError("Input size and the length of active dimensionals should be equal.")
        self.input_dim = input_dim
        self.active_dims = active_dims

        # convenient OrderedDict to make access to subkernels faster
        self._subkernels = OrderedDict()

    def forward(self, X, Z=None, diag=False):
        """
        Calculates covariance matrix of inputs on active dimensionals.

        :param torch.autograd.Variable X: A 2D tensor of size `N x input_dim`.
        :param torch.autograd.Variable Z: An optional 2D tensor of size `M x input_dim`.
        :param bool diag: A flag to decide if we want to return a full covariance matrix
            or just its diagonal part.
        :return: Covariance matrix of X and Z with size `N x M`.
        :rtype: torch.autograd.Variable
        """
        raise NotImplementedError

    def _slice_input(self, X):
        """
        Slices X according to `self.active_dims`. If X is 1 dimensional then returns
            a 2D tensor of size `N x 1`.

        :param torch.autograd.Variable X: A 1D or 2D tensor.
        :return: A 2D slice of X.
        :rtype: torch.autograd.Variable
        """
        if X.dim() == 2:
            return X[:, self.active_dims]
        elif X.dim() == 1:
            return X.unsqueeze(1)
        else:
            raise ValueError("Input X must be either 1 or 2 dimensional.")

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.mul(other)

    def __radd__(self, other):
        return self.add(other)

    def __rmul__(self, other):
        return self.mul(other)

    def add(self, other, name=None):
        """
        Returns a new kernel which acts like a sum/direct sum of ``self`` and ``other``.
        """
        return Sum(self, other, None)

    def mul(self, other, name=None):
        """
        Returns a new kernel which acts like a product/tensor product of ``self`` and ``other``.
        """
        return Product(self, other, name)

    def exp(self, name=None):
        """
        Creates a new kernel, which is an instance of Exponent kernel, according to:
        ``k_new(x, z) = exp(k(x, z))``.
        """
        return Exponent(self, name=name)

    def vertical_scale(self, vscaling_fn, name=None):
        """
        Creates a new kernel, which is an instance of VerticalScaling kernel, according to:
        ``k_new(x, z) = f(x)exp(k(x, z))f(z)``,
        where ``f := vscaling_fn`` is a vertical scaling function.
        """
        return VerticalScaling(self, vscaling_fn, name=name)

    def warp(self, iwarping_fn=None, owarping_coef=None, name=None):
        """
        Creates a new kernel, which is an instance of Warping kernel, according to:
        ``k_new(x, z) = q(k(f(x), f(z)))``,
        where ``f := iwarping_fn`` is a input warping function, and ``q`` is a polynomial
        with non-negative coefficients ``owarping_coef``.
        """
        return Warping(self, iwarping_fn, owarping_coef, name)

    def get_subkernel(self, name):
        """
        Returns the subkernel corresponding to ``name``.
        """
        if name in self._subkernels:
            return self._subkernels[name]
        else:
            raise KeyError("There is no subkernel with the name '{}'.".format(name))


class Combination(Kernel):
    def __init__(self, kernel0, kernel1, name=None):
        if not (isinstance(kernel0, Kernel) and
                (isinstance(kernel1, Kernel) or isinstance(kernel1, numbers.Number))):
            raise TypeError("Sub-kernels of a combined kernel must be a Kernel instance or a number.")

        active_dims = set(kernel0.active_dims)
        if isinstance(kernel1, Kernel):
            active_dims1 = set(kernel1.active_dims)
            if active_dims == active_dims1:  # on the same active_dims
                pass
            elif len(active_dims & active_dims1) == 0:  # on disjoint active_dims
                active_dims = active_dims + active_dims1
            else:
                raise ValueError("Sub-kernels must act on the same active dimensions or disjoint active "
                                 "dimensions (to create direct sum or tensor product or kernels).")

        active_dims = sorted(active_dims)
        input_dim = len(active_dims)
        super(Combination, self).__init__(input_dim, active_dims, name)

        self.kernel0 = kernel0
        self.kernel1 = kernel1

        if kernel0._subkernels:
            self._subkernels.update(kernel0._subkernels)
        else:
            self._subkernels[kernel0.name] = kernel0

        if isinstance(kernel1, Kernel):
            if kernel1._subkernels:
                subkernels1 = kernel1._subkernels
            else:
                subkernels1 = OrderedDict({kernel1.name: kernel1})
            for name, kernel in subkernels1.items():
                if name in self._subkernels:
                    raise KeyError("Detect two subkernels with the same name '{}'. "
                                   "Consider to change the default name of these subkernels "
                                   "to distinguish them.".format(name))
                self._subkernels[name] = kernel


class Sum(Combination):

    def forward(self, X, Z=None, diag=False):
        if isinstance(self.kernel1, Kernel):
            return self.kernel0(X, Z, diag) + self.kernel1(X, Z, diag)
        else:  # constant
            return self.kernel0(X, Z, diag) + self.kernel1


class Product(Combination):

    def forward(self, X, Z=None, diag=False):
        if isinstance(self.kernel1, Kernel):
            return self.kernel0(X, Z, diag) * self.kernel1(X, Z, diag)
        else:  # constant
            return self.kernel0(X, Z, diag) * self.kernel1


class Deriving():
    pass


class Warping(Deriving):
    pass


class Exponent(Deriving):
    pass


class VerticalScaling(Deriving):
    pass
