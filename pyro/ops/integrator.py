from __future__ import absolute_import, division, print_function

from torch.autograd import grad


def velocity_verlet(z, r, potential_fn, step_size, num_steps=1):
    """
    Second order symplectic integrator that uses the velocity verlet algorithm.

    :param dict z: dictionary of sample site names and their current values
        (type :class:`~torch.Tensor`).
    :param dict r: dictionary of sample site names and corresponding momenta
        (type :class:`~torch.Tensor`).
    :param callable potential_fn: function that returns potential energy given z
        for each sample site. The negative gradient of the function with respect
        to ``z`` determines the rate of change of the corresponding sites'
        momenta ``r``.
    :param float step_size: step size for each time step iteration.
    :param int num_steps: number of discrete time steps over which to integrate.
    :return tuple (z_next, r_next): final position and momenta, having same types as (z, r).
    """
    z_next = z.copy()
    r_next = r.copy()
    grads, _ = _grad(potential_fn, z_next)

    for _ in range(num_steps):
        for site_name in z_next:
            # r(n+1/2)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
            # z(n+1)
            z_next[site_name] = z_next[site_name] + step_size * r_next[site_name]
        grads, _ = _grad(potential_fn, z_next)
        for site_name in r_next:
            # r(n+1)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
    return z_next, r_next


def single_step_velocity_verlet(z, r, potential_fn, step_size, z_grads=None):
    """
    A special case of ``velocity_verlet`` integrator where ``num_steps=1``. It is particular
    helpful for NUTS kernel.

    :param torch.Tensor z_grads: optional gradients of potential energy at current ``z``.
    :return tuple (z_next, r_next, z_grads, potential_energy): next position and momenta,
        together with the potential energy and its gradient w.r.t. ``z_next``.
    """
    z_next = z.copy()
    r_next = r.copy()
    grads = _grad(potential_fn, z_next)[0] if z_grads is None else z_grads

    for site_name in z_next:
        r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
        z_next[site_name] = z_next[site_name] + step_size * r_next[site_name]
    grads, potential_energy = _grad(potential_fn, z_next)
    for site_name in r_next:
        r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
    return z_next, r_next, grads, potential_energy


def _grad(potential_fn, z):
    z_keys, z_nodes = zip(*z.items())
    for node in z_nodes:
        node.requires_grad = True
    potential_energy = potential_fn(z)
    grads = grad(potential_energy, z_nodes)
    for node in z_nodes:
        node.requires_grad = False
    return dict(zip(z_keys, grads)), potential_energy
