from __future__ import absolute_import, division, print_function

import numbers
from collections import OrderedDict

from pyro.contrib.gp.util import Parameterized


class Kernel(Parameterized):
    """
    Base class for kernels used in this Gaussian Process module.

    Every inherited class should implement a :meth:`forward` pass which takes inputs
    :math:`X`, :math:`Z` and returns their covariance matrix.

    To construct a new kernel from the old ones, we can use methods :meth:`add`,
    :meth:`mul`, :meth:`exp`, :meth:`warp`, :meth:`vertical_scale`.

    References:

    [1] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param int input_dim: Number of feature dimensions of inputs.
    :param torch.Tensor variance: Variance parameter of this kernel.
    :param list active_dims: List of feature dimensions of the input which the kernel
        acts on.
    :param str name: Name of the kernel.
    """

    def __init__(self, input_dim, active_dims=None, name=None):
        super(Kernel, self).__init__(name)

        if active_dims is None:
            active_dims = list(range(input_dim))
        elif input_dim != len(active_dims):
            raise ValueError("Input size and the length of active dimensionals should "
                             "be equal.")
        self.input_dim = input_dim
        self.active_dims = active_dims

        # convenient OrderedDict to make access to subkernels faster
        self._subkernels = OrderedDict()

    def forward(self, X, Z=None, diag=False):
        r"""
        Calculates covariance matrix of inputs on active dimensionals.

        :param torch.Tensor X: A 2D tensor with shape :math:`N \times input\_dim`.
        :param torch.Tensor Z: An (optional) 2D tensor with shape
            :math:`M \times input\_dim`.
        :param bool diag: A flag to decide if we want to return full covariance matrix
            or just its diagonal part.
        :returns: covariance matrix of :math:`X` and :math:`Z` with shape
            :math:`N \times M`
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _slice_input(self, X):
        """
        Slices :math:`X` according to ``self.active_dims``. If ``X`` is 1D then returns
        a 2D tensor with shape :math:`N \times 1`.

        :param torch.Tensor X: A 1D or 2D input tensor.
        :returns: a 2D slice of :math:`X`
        :rtype: torch.Tensor
        """
        if X.dim() == 2:
            return X[:, self.active_dims]
        elif X.dim() == 1:
            return X.unsqueeze(1)
        else:
            raise ValueError("Input X must be either 1 or 2 dimensional.")

    def add(self, other, name=None):
        """
        Creates a new kernel from a sum/direct sum of this kernel object and ``other``.

        :param other: A kernel to be combined with this kernel object.
        :type other: Kernel or numbers.Number
        :param str name: An optional name for the derived kernel.
        :returns: a Sum kernel
        :rtype: Sum
        """
        return Sum(self, other, name)

    def mul(self, other, name=None):
        """
        Creates a new kernel from a product/tensor product of this kernel object and
        ``other``.

        :param Kernel other: A kernel to be combined with this kernel object.
        :type other: Kernel or numbers.Number
        :param str name: An optional name for the derived kernel.
        :returns: a Product kernel
        :rtype: Product
        """
        return Product(self, other, name)

    def exp(self, name=None):
        """
        Creates a new kernel according to

            :math:`k_{new}(x, z) = \exp(k(x, z)).`

        :param str name: An optional name for the derived kernel.
        :returns: an Exponent kernel
        :rtype: Exponent
        """
        return Exponent(self, name)

    def vertical_scale(self, vscaling_fn, name=None):
        """
        Creates a new kernel according to

            :math:`k_{new}(x, z) = f(x)k(x, z)f(z),`

        where :math:`f` is a function.

        :param callable vscaling_fn: A vertical scaling function :math:`f`.
        :param str name: An optional name for the derived kernel.
        :returns: a vertical scaled kernel
        :rtype: VerticalScaling
        """
        return VerticalScaling(self, vscaling_fn, name)

    def warp(self, iwarping_fn=None, owarping_coef=None, name=None):
        """
        Creates a new kernel according to

            :math:`k_{new}(x, z) = q(k(f(x), f(z))),`

        where :math:`f` is an function and :math:`q` is a polynomial with non-negative
        coefficients ``owarping_coef``.

        :param callable iwarping_fn: An input warping function :math:`f`.
        :param list owarping_coef: A list of coefficients of the output warping
            polynomial. These coefficients must be non-negative.
        :param str name: An optional name for the derived kernel.
        :returns: a warped kernel
        :rtype: Warping
        """
        return Warping(self, iwarping_fn, owarping_coef, name)

    def get_subkernel(self, name):
        """
        Returns the subkernel corresponding to ``name``.

        :param str name: Name of the subkernel.
        :returns: A subkernel.
        :rtype: Kernel
        """
        if name in self._subkernels:
            return self._subkernels[name]
        else:
            raise KeyError("There is no subkernel with the name '{}'.".format(name))


class Combination(Kernel):
    """
    Base class for kernels derived from a combination of kernels.

    :param Kernel kern0: First kernel to combine.
    :param kern1: Second kernel to combine.
    :type kern1: Kernel or numbers.Number
    """
    def __init__(self, kern0, kern1, name=None):
        if not isinstance(kern0, Kernel):
            raise TypeError("The first component of a combined kernel must be a "
                            "Kernel instance.")
        if not (isinstance(kern1, Kernel) or isinstance(kern1, numbers.Number)):
            raise TypeError("The second component of a combined kernel must be a "
                            "Kernel instance or a number.")

        active_dims = set(kern0.active_dims)
        if isinstance(kern1, Kernel):
            active_dims |= set(kern1.active_dims)
        active_dims = sorted(active_dims)
        input_dim = len(active_dims)
        super(Combination, self).__init__(input_dim, active_dims, name)

        self.kern0 = kern0
        self.kern1 = kern1

        if kern0._subkernels:
            self._subkernels.update(kern0._subkernels)
        else:
            self._subkernels[kern0.name] = kern0

        if isinstance(kern1, Kernel):
            if kern1._subkernels:
                subkernels1 = kern1._subkernels
            else:
                subkernels1 = OrderedDict({kern1.name: kern1})
            for name, kernel in subkernels1.items():
                if name in self._subkernels:
                    if self._subkernels[name] is not kernel:
                        raise KeyError("Detect two different subkernels with the same "
                                       "name '{}'. Consider to change the default "
                                       "name of these subkernels to distinguish them."
                                       .format(name))
                else:
                    self._subkernels[name] = kernel


class Sum(Combination):
    """
    Returns a new kernel which acts like a sum/direct sum of two kernels.
    The second kernel can be a constant.
    """
    def forward(self, X, Z=None, diag=False):
        if isinstance(self.kern1, Kernel):
            return self.kern0(X, Z, diag) + self.kern1(X, Z, diag)
        else:  # constant
            return self.kern0(X, Z, diag) + self.kern1


class Product(Combination):
    """
    Returns a new kernel which acts like a product/tensor product of two kernels.
    The second kernel can be a constant.
    """
    def forward(self, X, Z=None, diag=False):
        if isinstance(self.kern1, Kernel):
            return self.kern0(X, Z, diag) * self.kern1(X, Z, diag)
        else:  # constant
            return self.kern0(X, Z, diag) * self.kern1


class Transforming(Kernel):
    """
    Base class for kernels derived from a kernel by some transforms such as warping,
    exponent, vertical scaling.

    :param Kernel kern: The original kernel.
    """
    def __init__(self, kern, name=None):
        super(Transforming, self).__init__(kern.input_dim, kern.active_dims, name)

        self.kern = kern

        if kern._subkernels:
            self._subkernels.update(kern._subkernels)
        else:
            self._subkernels[kern.name] = kern


class Exponent(Transforming):
    """
    Creates a new kernel according to

        :math:`k_{new}(x, z) = \exp(k(x, z)).`
    """
    def forward(self, X, Z=None, diag=False):
        return self.kern(X, Z, diag).exp()


class VerticalScaling(Transforming):
    """
    Creates a new kernel according to

        :math:`k_{new}(x, z) = f(x)k(x, z)f(z),`

    where :math:`f` is a function.

    :param callable vscaling_fn: A vertical scaling function :math:`f`.
    """
    def __init__(self, kern, vscaling_fn, name=None):
        super(VerticalScaling, self).__init__(kern, name)

        self.vscaling_fn = vscaling_fn

    def forward(self, X, Z=None, diag=False):
        if diag:
            return self.vscaling_fn(X) * self.kern(X, Z, diag) * self.vscaling_fn(X)
        elif Z is None:
            vscaled_X = self.vscaling_fn(X).unsqueeze(1)
            return vscaled_X * self.kern(X, Z, diag) * vscaled_X.t()
        else:
            return (self.vscaling_fn(X).unsqueeze(1) * self.kern(X, Z, diag) *
                    self.vscaling_fn(Z).unsqueeze(0))


def _Horner_evaluate(x, coef):
    """
    Evaluates the value of a polynomial according to Horner's method.
    """
    # https://en.wikipedia.org/wiki/Horner%27s_method
    n = len(coef) - 1
    b = coef[n]
    for i in range(n-1, -1, -1):
        b = coef[i] + b * x
    return b


class Warping(Transforming):
    """
    Creates a new kernel according to

        :math:`k_{new}(x, z) = q(k(f(x), f(z))),`

    where :math:`f` is an function and :math:`q` is a polynomial with non-negative
    coefficients ``owarping_coef``.

    We can take advantage of :math:`f` to combine a Gaussian Process kernel with a deep
    learning architecture. For example:

        >>> linear = torch.nn.Linear(10, 3)
        >>> # register its parameters to Pyro's ParamStore and wrap it by lambda
        >>> # to call the primitive pyro.module each time we use the linear function
        >>> pyro_linear_fn = lambda x: pyro.module("linear", linear)(x)
        >>> kernel = gp.kernels.Matern52(input_dim=3, lengthscale=torch.ones(3))
        >>> warped_kernel = gp.kernels.Warping(kernel, pyro_linear_fn)

    Reference:

    [1] `Deep Kernel Learning`,
    Andrew G. Wilson, Zhiting Hu, Ruslan Salakhutdinov, Eric P. Xing

    :param callable iwarping_fn: An input warping function :math:`f`.
    :param list owarping_coef: A list of coefficients of the output warping polynomial.
        These coefficients must be non-negative.
    """
    def __init__(self, kern, iwarping_fn=None, owarping_coef=None, name=None):
        super(Warping, self).__init__(kern, name)

        self.iwarping_fn = iwarping_fn

        if owarping_coef is not None:
            for coef in owarping_coef:
                if not isinstance(coef, int) and coef < 0:
                    raise ValueError("Coefficients of the polynomial must be a "
                                     "non-negative integer.")
            if len(owarping_coef) < 2 and sum(owarping_coef) == 0:
                raise ValueError("The ouput warping polynomial should have a degree "
                                 "of at least 1.")
        self.owarping_coef = owarping_coef

    def forward(self, X, Z=None, diag=False):
        if self.iwarping_fn is None:
            K_iwarp = self.kern(X, Z, diag)
        elif Z is None:
            K_iwarp = self.kern(self.iwarping_fn(X), None, diag)
        else:
            K_iwarp = self.kern(self.iwarping_fn(X), self.iwarping_fn(Z), diag)

        if self.owarping_coef is None:
            return K_iwarp
        else:
            return _Horner_evaluate(K_iwarp, self.owarping_coef)
