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


def gaussian_contract_base(yz, xy, nx):
    """
    Computes Gaussian contract of yz, xy assuming `x` disjoint `z`.

    :param yz: Gaussian over y, z
    :param xy: Gaussian over x, y
    :param nx: size of x
    """
    # TODO: remove these legacy computations if they are unnecessary
    # p(a,b)q(a,c) = p(a)p(b|a)q(c)q(b|c)
    # p(b|a) = N(b; mx.b + inv(Px.bb) @ Px.ba @ (a - mx.a), inv(Px.bb))
    # q(b|c) = N(b; my.b + inv(Py.bb) @ Py.bc @ (c - my.c), inv(Py.bb))
    # p(b|a)q(b|c) = C * N(b; mb, inv(Pb))
    # mb = inv(Px.bb + Py.bb) @ (Px.bb @ mx.b + Px.ba @ (a - mx.a)
    #                                    Py.bb @ my.b + Py.bc @ (c - my.c))
    # Pb = Px.bb + Py.bb
    # C = N(0; mx.b - my.b + inv(Px.bb) @ Px.ba @ (a - mx.a) - inv(Py.bb) @ Py.bc @ (c - my.c),
    #          inv(Px.bb) + inv(Py.bb))
    # Cp(a)q(c) = ...
    # It seems complicated to represent in terms of N((a, c); mz.a, mz.c, ...)
    ny = xy.size(-1) - nx
    Ayz, byz, Cyz, etayz, Jyz = mean_precision_to_filtering_parameters(yz.mean, yz.precision, ny)
    Axy, bxy, Cxy, etaxy, Jxy = mean_precision_to_filtering_parameters(xy.mean, xy.precision, nx)
    # compute inv(I + Jxy @ Cyz)
    # TODO: merge terms and use a unique triangular_solve
    tmp = torch.inverse(torch.matmul(Jxy, Cyz) + torch.eye(ny))
    Axy_tmp = torch.matmul(Axy, tmp)
    A = torch.matmul(Axy_tmp, Ayz)
    b = torch.matmul(Axy_tmp, b + torch.matmul(Cyz, etaxy.unsqueeze(-1).squeeze(-1))) + bxy
    C = torch.matmul(Axy_tmp, torch.matmul(Cyz, torch.transpose(Ayz, -2, -1))) + Cxy
    Ayzt_tmp = torch.matmul(torch.transpose(Ayz, -2, -1), tmp)
    eta = torch.matmul(Ayzt_tmp, etaxy - torch.matmul(Jxy, byz)) + etayz
    J = torch.matmul(Ayzt_tmp, torch.matmul(Jxy, Ayz)) + Jyz
    mean, precision = filtering_parameters_to_mean_precision(A, b, C, eta, J)
    return Gaussian(0, mean, precision)


def mean_precision_to_filtering_parameters(mean, precision, nx):
    """
    Transforms (mean, precision) to (A, b, C, eta, J), where

        p(x, z) = p(x | z) p(z) = N(x; Az + b, C) N(z; inv(J)eta, inv(J))

    :param mean: mean of Gaussian
    :param precision: precision of Gaussian
    :param nx: size of x
    """
    # TODO: explore precision_tril alternative
    # TODO: use ellipsis for batching
    mx, mz = mean[:i], mean[i:]
    Pxx, Pxz, Pzx, Pzz = precision[:i, :i], precision[:i, i:], precision[i:, :i], precision[i:, i:]
    # p(x | z) = N(x; mx - inv(Pxx) @ Pxz @ (z - mz), inv(Pxx))
    C = torch.invert(Pxx)
    A = -torch.matmul(C, Pxz)
    b = mx - torch.matmul(A, mz.unsqueeze(-1)).squeeze(-1)
    # p(z) = N(z; mz, Czz) = N(z; mz, inv(Pzz - Pzx @ inv(Pxx) @ Pxz))
    J = Pzz + torch.matmul(Pzx, A)
    eta = torch.matmul(J, mz.unsqueeze(-1)).squeeze(-1)
    return A, b, C, eta, J


def filtering_parameters_to_mean_precision(A, b, C, eta, J):
    # inverse of `mean_precision_to_filtering_parameters` function
    Pxx = torch.invert(C)
    Pxz = -torch.matmul(Pxx, A)
    Pzx = torch.transpose(Pxz, -2, -1)
    Pzz = J - torch.matmul(Pzx, A)
    mz = torch.solve(J, eta.unsqueeze(-1)).squeeze(-1)
    mx = b + torch.matmul(A, mz.unsqueeze(-1)).squeeze(-1)
    mean = torch.cat([mx, mz], dim=-1)
    precision = torch.cat([torch.cat([Pxx, Pxz], dim=-1), torch.cat([Pzx, Pzz], dim=-1)], dim=-2)
    return mean, precision


def gaussian_contract(equation, x, y):
    """
    Compute the integral over two gaussians:

        (x @ y)(a,c) = log(integral(exp(x(a,b) + y(b,c)), b))

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
