from __future__ import absolute_import, division, print_function


def _matrix_triangular_solve_compat(b, A, upper=True):
    """Computes the solution to the linear equation AX = b, where A is a triangular matrix."""
    if A.requires_grad or A.is_cuda:
        return A.inverse().matmul(b)
    else:
        return b.trtrs(A, upper=upper)[0].view(b.size())
