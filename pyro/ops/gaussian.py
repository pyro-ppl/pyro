import torch

from pyro.distributions.util import broadcast_shape


class Gaussian(object):
    """
    Non-normalized Gaussian distribution.

    This represents an arbitrary semidefinite quadratic function, which can be
    interpreted as a rank-deficient scaled Gaussian distribution. The precision
    matrix may have zero eigenvalues, thus it may be impossible to work
    directly with the covariance matrix.

    TODO(fehiepsi) Possibly change precision to another more numerically stable
    representation.
    """
    def __init__(self, log_normalizer, mean, precision):
        assert mean.dim() >= 1
        assert precision.dim() >= 2
        assert precision.shape[-2:] == mean.shape[-1:] * 2
        self.log_normalizer = log_normalizer
        self.mean = mean
        self.precision = precision

    def dim(self):
        return self.mean.size(-1)

    @property
    def batch_shape(self):
        return broadcast_shape(self.log_normalizer.shape,
                               self.mean.shape[:-1],
                               self.precision.shape[:-2])

    def log_density(self, value):
        """
        Evaluate the log density of this Gaussian at a point value.
        This is mainly used for testing.
        """
        diff = value - self.mean
        result = torch.matmul(diff.unsqueeze(-2), self.precision)
        result = torch.matmul(result, diff.unsqueeze(-1))
        return result.sum(-1).sum(-1) + self.log_normalizer


def gaussian_contract(equation, x, y):
    """
    Compute the integral over two gaussians:

        (x @ y)(a,c) = log(integral(exp(x(a,b) + y(b,c)), c))

    where x is a gaussian over variables a,b, y is a gaussian over variables
    b,c, and a,b,c can each be sets of zero or more variables.
    """
    assert isinstance(x, Gaussian)
    assert isinstance(y, Gaussian)
    inputs, output = equation.split("->")
    x_input, y_input = inputs.split(",")
    assert set(output) <= set(x_input + y_input)
    assert len(x_input) == x.dim()
    assert len(y_input) == y.dim()

    # TODO(fehiepsi) Compute fused gaussian.
    raise NotImplementedError("TODO")
    result = "TODO"

    # Sketch of precision computation:
    # full_precision = (x.precision.pad(...) + y.precision.pad(...))
    # full_cov = torch.inv(full_precision)
    # cov = full_cov[selected_indices, selected_indices]
    # precision = torch.inv(cov)

    assert len(output) == result.dim()
    return result
