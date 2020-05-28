# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch.autograd import grad


def velocity_verlet(z, r, potential_fn, kinetic_grad, step_size, num_steps=1, z_grads=None):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm.

    :param dict z: dictionary of sample site names and their current values
        (type :class:`~torch.Tensor`).
    :param dict r: dictionary of sample site names and corresponding momenta
        (type :class:`~torch.Tensor`).
    :param callable potential_fn: function that returns potential energy given z
        for each sample site. The negative gradient of the function with respect
        to ``z`` determines the rate of change of the corresponding sites'
        momenta ``r``.
    :param callable kinetic_grad: a function calculating gradient of kinetic energy
        w.r.t. momentum variable.
    :param float step_size: step size for each time step iteration.
    :param int num_steps: number of discrete time steps over which to integrate.
    :param torch.Tensor z_grads: optional gradients of potential energy at current ``z``.
    :return tuple (z_next, r_next, z_grads, potential_energy): next position and momenta,
        together with the potential energy and its gradient w.r.t. ``z_next``.
    """
    z_next = z.copy()
    r_next = r.copy()
    for _ in range(num_steps):
        z_next, r_next, z_grads, potential_energy = _single_step_verlet(z_next,
                                                                        r_next,
                                                                        potential_fn,
                                                                        kinetic_grad,
                                                                        step_size,
                                                                        z_grads)
    return z_next, r_next, z_grads, potential_energy


def _single_step_verlet(z, r, potential_fn, kinetic_grad, step_size, z_grads=None):
    r"""
    Single step velocity verlet that modifies the `z`, `r` dicts in place.
    """

    z_grads = potential_grad(potential_fn, z)[0] if z_grads is None else z_grads

    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (-z_grads[site_name])  # r(n+1/2)

    r_grads = kinetic_grad(r)
    for site_name in z:
        z[site_name] = z[site_name] + step_size * r_grads[site_name]  # z(n+1)

    z_grads, potential_energy = potential_grad(potential_fn, z)
    for site_name in r:
        r[site_name] = r[site_name] + 0.5 * step_size * (-z_grads[site_name])  # r(n+1)

    return z, r, z_grads, potential_energy


def potential_grad(potential_fn, z):
    """
    Gradient of `potential_fn` w.r.t. parameters z.

    :param potential_fn: python callable that takes in a dictionary of parameters
        and returns the potential energy.
    :param dict z: dictionary of parameter values keyed by site name.
    :return: tuple of `(z_grads, potential_energy)`, where `z_grads` is a dictionary
        with the same keys as `z` containing gradients and potential_energy is a
        torch scalar.
    """
    z_keys, z_nodes = zip(*z.items())
    for node in z_nodes:
        node.requires_grad_(True)
    try:
        potential_energy = potential_fn(z)
    # deal with singular matrices
    except RuntimeError as e:
        if "singular U" in str(e):
            grads = {k: v.new_zeros(v.shape) for k, v in z.items()}
            return grads, z_nodes[0].new_tensor(float('nan'))
        else:
            raise e

    grads = grad(potential_energy, z_nodes)
    for node in z_nodes:
        node.requires_grad_(False)
    return dict(zip(z_keys, grads)), potential_energy.detach()
