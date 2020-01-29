# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0


class DualAveraging:
    """
    Dual Averaging is a scheme to solve convex optimization problems. It belongs
    to a class of subgradient methods which uses subgradients to update parameters
    (in primal space) of a model. Under some conditions, the averages of generated
    parameters during the scheme are guaranteed to converge to an optimal value.
    However, a counter-intuitive aspect of traditional subgradient methods is
    "new subgradients enter the model with decreasing weights" (see :math:`[1]`).
    Dual Averaging scheme solves that phenomenon by updating parameters using
    weights equally for subgradients (which lie in a dual space), hence we have
    the name "dual averaging".

    This class implements a dual averaging scheme which is adapted for Markov chain
    Monte Carlo (MCMC) algorithms. To be more precise, we will replace subgradients
    by some statistics calculated during an MCMC trajectory. In addition,
    introducing some free parameters such as ``t0`` and ``kappa`` is helpful and
    still guarantees the convergence of the scheme.

    References

    [1] `Primal-dual subgradient methods for convex problems`,
    Yurii Nesterov

    [2] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, Andrew Gelman

    :param float prox_center: A "prox-center" parameter introduced in :math:`[1]`
        which pulls the primal sequence towards it.
    :param float t0: A free parameter introduced in :math:`[2]`
        that stabilizes the initial steps of the scheme.
    :param float kappa: A free parameter introduced in :math:`[2]`
        that controls the weights of steps of the scheme.
        For a small ``kappa``, the scheme will quickly forget states
        from early steps. This should be a number in :math:`(0.5, 1]`.
    :param float gamma: A free parameter which controls the speed
        of the convergence of the scheme.
    """

    def __init__(self, prox_center=0, t0=10, kappa=0.75, gamma=0.05):
        self.prox_center = prox_center
        self.t0 = t0
        self.kappa = kappa
        self.gamma = gamma
        self.reset()

    def reset(self):
        self._x_avg = 0  # average of primal sequence
        self._g_avg = 0  # average of dual sequence
        self._t = 0

    def step(self, g):
        """
        Updates states of the scheme given a new statistic/subgradient ``g``.

        :param float g: A statistic calculated during an MCMC trajectory or subgradient.
        """
        self._t += 1
        # g_avg = (g_1 + ... + g_t) / t
        self._g_avg = (1 - 1/(self._t + self.t0)) * self._g_avg + g / (self._t + self.t0)
        # According to formula (3.4) of [1], we have
        #     x_t = argmin{ g_avg . x + loc_t . |x - x0|^2 },
        # where loc_t := beta_t / t, beta_t := (gamma/2) * sqrt(t)
        self._x_t = self.prox_center - (self._t ** 0.5) / self.gamma * self._g_avg
        # weight for the new x_t
        weight_t = self._t ** (-self.kappa)
        self._x_avg = (1 - weight_t) * self._x_avg + weight_t * self._x_t

    def get_state(self):
        r"""
        Returns the latest :math:`x_t` and average of
        :math:`\left\{x_i\right\}_{i=1}^t` in primal space.
        """
        return self._x_t, self._x_avg
