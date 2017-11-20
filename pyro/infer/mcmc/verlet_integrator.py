from torch.autograd import Variable


def verlet_integrator(z, r, grad_potential, step_size, num_steps):
    """
    Velocity Verlet integrator.

    :param z: dictionary of sample site names and their current values
    :param r: dictionary of sample site names and corresponding momenta
    :param grad_potential: function that returns gradient of the potential given z
        for each sample site
    :return: (z_next, r_next) having same types as (z, r)
    """
    # deep copy the current state - (z, r)
    z_next = {key: val.clone().detach() for key, val in z.items()}
    r_next = {key: val.clone().detach() for key, val in r.items()}
    retain_grads(z_next)
    grads = grad_potential(z_next)

    for _ in range(num_steps):
        # detach graph nodes for next iteration
        detach_nodes(z_next)
        detach_nodes(r_next)
        for site_name in z_next:
            # r(n+1/2)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
            # z(n+1)
            z_next[site_name] = z_next[site_name] + step_size * r_next[site_name]
        # retain gradients for intermediate nodes in backward step
        retain_grads(z_next)
        grads = grad_potential(z_next)
        for site_name in r_next:
            # r(n+1)
            r_next[site_name] = r_next[site_name] + 0.5 * step_size * (-grads[site_name])
    return z_next, r_next


def retain_grads(z):
    for value in z.values():
        # XXX: can be removed with PyTorch 0.3
        if value.is_leaf and not value.requires_grad:
            value.requires_grad = True
        value.retain_grad()


def detach_nodes(z):
    for key, value in z.items():
        z[key] = Variable(value.data, requires_grad=True)