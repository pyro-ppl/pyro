from __future__ import absolute_import, division, print_function


class DualAveraging(object):
    """
    Dual Averaging is a scheme to solve convex optimization problems. It belongs to a class of subgradient
    methods which uses subgradients to update parameters (in primal space) of a model. Under some conditions,
    the averages of generated parameters during the scheme are guaranteed to converge to an optimal value.
    However, a counter-intuitive aspect of traditional subgradient methods is "new subgradients enter the
    model with decreasing weights". Dual Averaging scheme solves that phenomenon by updating parameters using
    weights equally for subgradients (which lie in a dual space), hence we have the name "dual averaging".

    This class implements a dual averaging scheme which is adapted for Markov chain Monte Carlo (MCMC)
    algorithms. To be more precise, we will replace subgradients by some statistics calculated during an
    MCMC trajectory. In addition, introducing some free parameters such as ``t0`` and ``kappa``is helpful
    and still guarantees the convergence of the scheme.

    References

    [1] `Primal-dual subgradient methods for convex problems`,
    Yurii Nesterov

    [2] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, Andrew Gelman

    :param float x0: A "prox-center" parameter introduced in :math:`[1]` which pulls the primal sequence towards it.
    :param float t0: A free parameter introduced in :math:`[2]` that stabilizes the initial steps of the scheme.
    :param float kappa: A free parameter introduced in :math:`[2]` that controls the weights of steps of the scheme.
        For a small ``kappa``, the scheme will quickly forget states from early steps.
        This should be a number in :math:`(0.5, 1]`.
    :param float gamma: A free parameter which controls the speed of the convergence of the scheme.
    """

    def __init__(self, x0=0, t0=10, kappa=0.75, gamma=0.05):
        self.x0 = x0
        self.t0 = t0
        self.kappa = kappa
        self.gamma = gamma

        self._x_avg = 0  # average of primal sequence
        self._g_avg = 0  # average of dual sequence
        self._t = 0

    def step(self, g):
        """
        Updates states of the scheme given a new subgradient/statistics ``g``.

        :param float g: New statistics calculated during an MCMC trajectory.
        """
        self._t += 1
        # g_avg = (g_1 + ... + g_t) / t
        self._g_avg = (1 - 1/(self._t + self.t0)) * self._g_avg + g / (self._t + self.t0)
        # According to formula (3.4) of [1], we have
        #     x_t = argmin{ g_avg . x + mu_t . |x - x0|^2 },
        # where mu_t := beta_t / t, beta_t := (gamma/2) * sqrt(t)
        x_t = self.x0 - (self._t ** 0.5) / self.gamma * self._g_avg
        # weight for the new x_t
        weight_t = self._t ** (-self.kappa)
        self._x_avg = (1 - weight_t) * self._x_avg + weight_t * x_t

    def get_state(self):
        r"""
        Returns the avarage of :math:`\left\{x_t\right\}` in primal space
        and the avarage of :math:`\left\{g_t\right\}` in dual space.
        """
        return self._x_avg, self._g_avg
