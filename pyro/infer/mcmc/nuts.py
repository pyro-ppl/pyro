from __future__ import absolute_import, division, print_function

from collections import namedtuple

import torch

import pyro
import pyro.distributions as dist
from pyro.infer.util import torch_data_sum
from pyro.ops.integrator import velocity_verlet
from pyro.util import ng_ones, ng_zeros

from .hmc import HMC


_TreeInfo = namedtuple("TreeInfo", ["z_left", "r_left", "z_right", "r_right",
                                    "z_proposal", "size", "turning", "diverging"])


class NUTS(HMC):
    """
    No-U-Turn Sampler kernel, where ``step_size`` need to be explicitly specified by the user.

    References

    [1] `The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo`,
    Matthew D. Hoffman, and Andrew Gelman

    :param model: python callable containing pyro primitives.
    :param float step_size: determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics.
    """

    def __init__(self, model, step_size=0.5):
        super(NUTS, self).__init__(model, step_size, num_steps=None)
        self._reset()
        self._dE_max = 1000

    def _is_turning(self, z_left, r_left, z_right, r_right):
        nodes = sorted(z_left)
        z_left = torch.stack([z_left[name] for name in nodes])
        r_left = torch.stack([r_left[name] for name in nodes])
        z_right = torch.stack([z_right[name] for name in nodes])
        r_right = torch.stack([r_right[name] for name in nodes])
        dz = z_right - z_left
        return (torch_data_sum(dz * r_left) < 0) or (torch_data_sum(dz * r_right) < 0)

    def _build_basetree(self, z, r, log_slice):
        z_new, r_new = velocity_verlet(z, r, self._potential_energy, self.step_size)
        dE = (log_slice + self._energy(z_new, r_new)).data[0]
        tree_size = 1 if dE <= 0 else 0
        diverging = dE >= self._dE_max
        return _TreeInfo(z_new, r_new, z_new, r_new, z_new, tree_size, False, diverging)

    def _build_tree(self, z, r, log_slice, direction, tree_depth):
        if tree_depth == 0:
            return self._build_basetree(z, r, log_slice)

        # build half of tree
        half_tree = self._build_tree(z, r, log_slice, direction, tree_depth-1)
        if half_tree.turning or half_tree.diverging:
            return half_tree

        # else, build remaining half of tree
        if direction == 1:
            z = half_tree.z_right
            r = half_tree.r_right
        else:
            z = half_tree.z_left
            r = half_tree.r_left
        other_half_tree = self._build_tree(z, r, log_slice, direction, tree_depth-1)

        tree_size = half_tree.size + other_half_tree.size
        prob_other = other_half_tree.size / tree_size if tree_size != 0 else 1
        rand = int(dist.Bernoulli(ps=ng_ones(1) * prob_other)().data[0])
        z_proposal = other_half_tree.z_proposal if rand == 1 else half_tree.z_proposal

        if direction == 1:
            z_left = half_tree.z_left
            r_left = half_tree.r_left
            z_right = other_half_tree.z_right
            r_right = other_half_tree.r_right
        else:
            z_left = other_half_tree.z_left
            r_left = other_half_tree.r_left
            z_right = half_tree.z_right
            r_right = half_tree.r_right

        turning = self._is_turning(z_left, r_left, z_right, r_right) or other_half_tree.turning
        diverging = other_half_tree.diverging

        return _TreeInfo(z_left, r_left, z_right, r_right, z_proposal, tree_size, turning, diverging)

    def sample(self, trace):
        z = {name: node['value'] for name, node in trace.iter_stochastic_nodes()}
        r = {name: pyro.sample('r_{}_t={}'.format(name, self._t), self._r_dist[name]) for name in sorted(z)}
        slice_var = dist.Uniform(ng_zeros(1), torch.exp(-self._energy(z, r)))()
        log_slice = slice_var.log()
        z_left = z_right = z
        r_left = r_right = r
        tree_depth = 0
        tree_size = 1
        turning = diverging = False
        is_accept = False

        while not (turning or diverging):
            direction = pyro.sample("direction_t={}_depth={}".format(self._t, tree_depth),
                                    dist.Bernoulli(ps=ng_ones(1) * 0.5))
            direction = int(direction.data[0])
            if direction == 1:
                tree = self._build_tree(z_left, r_left, log_slice, direction, tree_depth)
                z_left = tree.z_left
                r_left = tree.r_left
            else:
                tree = self._build_tree(z_right, r_right, log_slice, direction, tree_depth)
                z_right = tree.z_right
                r_right = tree.r_right

            if not (tree.turning or tree.diverging):
                rand = pyro.sample("rand_t={}_depth={}".format(self._t, tree_depth),
                                   dist.Uniform(ng_zeros(1), ng_ones(1)))
                if rand.data[0] < tree.size / tree_size:
                    is_accept = True
                    z = tree.z_proposal

            tree_size += tree.size
            turning = self._is_turning(z_left, r_left, z_right, r_right) or tree.turning
            diverging = tree.diverging
            tree_depth += 1

        if is_accept:
            self._accept_cnt += 1
        self._t += 1
        return self._get_trace(z)
