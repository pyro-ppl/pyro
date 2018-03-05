from __future__ import absolute_import, division, print_function

from torch.autograd import Variable, grad


def unconstrained_velocity_verlet(z, r, potential_fn, step_size, num_steps):
    """
    Velocity verlet integrator for the case where all sample sites are unconstrained.
    See :func:`~pyro.ops.integrator.velocity_verlet`.
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


def single_step_velocity_verlet(z, r, potential_fn, step_size, z_grads=None, transforms={}):
    """
    A special case of ``velocity_verlet`` integrator where ``num_steps=1``. It is particular
    helpful for NUTS kernel.

    :param torch.Tensor z_grads: optional gradients of potential energy at current ``z``.
    :return tuple (z_next, r_next, z_grads, potential_energy): next position and momenta,
        together with the potential energy and its gradient w.r.t. ``z_next``.
    """
    # transform z to unconstrained space based on the specified transforms
    u_next = z.copy()
    for name, transform in transforms.items():
        u_next[name] = transform(u_next[name])
    r_next = r.copy()
    unconstrained_potential_fn = _unconstrained_potential_fn(potential_fn, transforms)
    grads = _grad(unconstrained_potential_fn, u_next)[0] if z_grads is None else z_grads

    for site_name in u_next:
        r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
        u_next[site_name] = u_next[site_name] + step_size * r_next[site_name]
    grads, model_potential = _grad(unconstrained_potential_fn, u_next)
    for site_name in r_next:
        r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])

    # return values in constrained space
    z_next = u_next
    for name, transform in transforms.items():
        z_next[name] = transform.inv(z[name])
    return z_next, r_next, grads, model_potential


def _grad(potential_fn, z):
    z = {k: Variable(v, requires_grad=True) for k, v in z.items()}
    z_keys, z_nodes = zip(*z.items())
    unconstrained_potential = potential_fn(z)
    grads = grad(unconstrained_potential, z_nodes)
    grads = [v.data for v in grads]
    return dict(zip(z_keys, grads)), unconstrained_potential


def _unconstrained_potential_fn(potential_fn, transforms={}):
    def _potential(u):
        z = u.copy()
        for name, transform in transforms.items():
            z[name] = transform.inv(u[name])
        model_potential = potential_fn(z)
        unconstrained_potential = model_potential.clone()
        for name, transform in transforms.items():
            unconstrained_potential += transform.log_abs_det_jacobian(z[name], u[name]).sum()
        return unconstrained_potential, model_potential

    return _potential


def velocity_verlet(z, r, potential_fn, step_size, num_steps, transforms={}):
    """
    Second order symplectic integrator that uses the velocity verlet algorithm.

    :param dict z: dictionary of sample site names and their current values
        (type ``torch.autograd.Variable``).
    :param dict r: dictionary of sample site names and corresponding momenta
        (type ``torch.autograd.Variable``).
    :param callable potential_fn: function that returns potential energy given z
        for each sample site. The negative gradient of the function with respect
        to ``z`` determines the rate of change of the corresponding sites'
        momenta ``r``.
    :param float step_size: step size for each time step iteration.
    :param int num_steps: number of discrete time steps over which to integrate.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
    :return tuple (z_next, r_next): final position and momenta, having same types as (z, r).
    """
    # transform z to unconstrained space based on the specified transforms
    u = z.copy()
    for name, transform in transforms.items():
        u[name] = transform(u[name])
    unconstrained_potential_fn = _unconstrained_potential_fn(potential_fn, transforms)
    u_next, r_next = unconstrained_velocity_verlet(u, r, unconstrained_potential_fn, step_size, num_steps)

    # return values in constrained space
    z_next = u_next
    for name, transform in transforms.items():
        z_next[name] = transform.inv(z_next[name])
    return z_next, r_next
