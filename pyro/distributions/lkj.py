from torch.distributions import Transform
from torch.distributions import constraints
import math

class LKJCholeskyTransform(Transform):
    domain = constraints.interval(-1, 1)
    codomain = constraints.lower_cholesky
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, LKJCholeskyTransform)

    def _call(self, y):
        D = (1.0 + math.sqrt(1.0 + 8.0 * y.shape[0]))/2.0
        if D % 1 != 0:
            raise ValueError("LKJ Priors require d choose 2 inputs")
        D = int(D)
        # Start with the identity matrix, to simplify things
        z = y.tanh()
        w = torch.eye(D, device=y.device)

        # compute y from the vector x as in Stan reference:
        # https://mc-stan.org/docs/2_18/reference-manual/correlation-matrix-transform-section.html
        last_z = z[0:(D-1)]
        w[1:D, 0] = last_z
        i = D - 1
        for d in range(1, D-1):
            distance_to_copy = D - 1 - d
            new_z = z[i:(i + distance_to_copy)]
            w[(d+1):D, d] = new_z * w[(d+1):, d - 1] * (1.0 - last_z[1:].pow(2)).sqrt()
            i += distance_to_copy
            last_z = new_z

        return w

    def _inverse(self, w):
        if (w.shape[0] != w.shape[1]):
            raise ValueError("A matrix that isn't square can't be a Cholesky factor of a correlation matrix")
        D = w.shape[0]

        def z_inverse(x):
            return (1 - x.pow(2)).sqrt()

        z_stack = [
            w[1:, 0]
        ]
        z_inv_stack = [
            z_inverse(z_stack[0])
        ]

        for d in range(1, D-1):
            z = w[(d+1):, d]
            for dd in range(0, d):
                old_z = z_inv_stack[dd][(d - dd):]
                old_w = w[(d+1):, dd]
                z /= (old_z * old_w)
            z_stack.append(z)
            z_inv_stack.append(z_inverse(z))

        x = torch.cat(z_stack)
        return torch.log((x + 1) / (1 - x))/2

    def log_abs_det_jacobian(self, x, y):
        # see the above reference
        return log_abs_det
