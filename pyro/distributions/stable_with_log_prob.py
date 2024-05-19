import torch
import math

from functools import partial

from scipy.special import roots_legendre

from .stable import Stable

def set_integrator(num_points):
    global integrate
    roots, weights = roots_legendre(num_points)
    roots = torch.Tensor(roots).double()
    weights = torch.Tensor(weights).double()
    log_weights = weights.log()
    half_roots = roots * 0.5
    def integrate(fn, domain):
        sl = [slice(None)] + (len(domain.shape) - 1) * [None]
        half_roots_sl = half_roots[sl]
        value = domain[0] * (0.5 - half_roots_sl) + domain[1] * (0.5 + half_roots_sl)
        return torch.logsumexp(fn(value) + log_weights[sl], dim=0) + ((domain[1] - domain[0]) / 2).log()


set_integrator(num_points=501)


class StableWithLogProb(Stable):
    def log_prob(self, value):
        # Undo shift and scale
        value = (value - self.loc) / self.scale

        # Use double precision math
        alpha = self.stability.double()
        beta = self.skew.double()
        value = value.double()

        # Optionally convert from Nolan's parametrization S^0 where samples depend
        # continuously on (alpha,beta), allowing interpolation around the hole at
        # alpha=1.
        if self.coords == "S0":
            value = value + beta * (math.pi / 2 * alpha).tan()
        elif self.coords != "S":
            raise ValueError("Unknown coords: {}".format(self.coords))

        # Integration is not valid for very small values
        value = torch.where(value > 0, value.clamp(min=0.01), value.clamp(max=-0.01))

        log_prob = _unsafe_stable_log_prob(alpha, beta, value)

        return log_prob - self.scale.log()


def _unsafe_stable_log_prob(alpha, beta, Z):
    # Calculate log-probability of Z given V. This will fail if alpha is close to 1
    # or if Z is close to 0
    ha = math.pi / 2 * alpha
    b = beta * ha.tan()
    atan_b = b.atan()
    u_zero = -alpha.reciprocal() * atan_b
    
    # If sample should be negative calculate with flipped beta and flipped value
    flip_beta_x = Z < 0
    beta = torch.where(flip_beta_x, -beta, beta)
    u_zero = torch.where(flip_beta_x, -u_zero, u_zero)
    Z = torch.where(flip_beta_x, -Z, Z)

    # Set integration domwin
    domain = torch.stack((u_zero, 0.5 * math.pi * u_zero.new_ones(u_zero.shape)), dim=0)

    integrand = partial(_unsafe_stable_given_uniform_log_prob, alpha=alpha, beta=beta, Z=Z)
    
    return integrate(integrand, domain) - math.log(math.pi)


def _unsafe_stable_given_uniform_log_prob(V, alpha, beta, Z):
    # Calculate log-probability of Z given V. This will fail if alpha is close to 1
    # or if Z is close to 0
    inv_alpha = alpha.reciprocal()
    half_pi = math.pi / 2
    eps = torch.finfo(V.dtype).eps
    # make V belong to the open interval (-pi/2, pi/2)
    V = V.clamp(min=2 * eps - half_pi, max=half_pi - 2 * eps)
    ha = half_pi * alpha
    b = beta * ha.tan()
    atan_b = b.atan()
    
    # +/- `ha` term to keep the precision of alpha * (V + half_pi) when V ~ -half_pi
    v = atan_b - ha + alpha * (V + half_pi)
    W = ( ( v.sin() / Z /
           (atan_b.cos() * V.cos()).pow(inv_alpha)
          ).pow(alpha / (1 - alpha))
          * (v - V).cos().clamp(min=eps) 
        )

    return -W + (alpha * W / Z / (alpha - 1)).abs().log()