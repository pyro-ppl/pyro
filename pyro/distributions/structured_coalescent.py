# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints


def _mv(matrix, vector):
    return matrix.matmul(vector.unsqueeze(-1)).squeeze(-1)


def _interpolate_mv(t0, t1, matrix, vector):
    if matrix.dim() == 2 or matrix.size(-3) == 1:
        # homogeneous
        matrix = matrix.matrix_power(t1 - t0)
        return _mv(matrix, vector)
    raise NotImplementedError("TODO")


def _structured_coalescent(
    time, parent, state_init, state_trans, coal_rate, *,
    validate_args=None,
):
    """

    **References:**

    [1] Peter Beerli, Joseph Felsenstein (2001)
        `Maximum likelihood estimation of a migration matrix and effective
        population sizes in n subpopulations by using a coalescent approach`
        https://www.pnas.org/content/98/8/4563
    [2] T. Vaughan, D. Kuhnert, A. Popinga, D. Welch, A. Drummond (2014)
        `Efficient Bayesian inference under the structured coalescent`
        https://academic.oup.com/bioinformatics/article/30/16/2272/2748160

    :param Tensor time: Maps node_id to the time of that node.
        This should be sorted in ascending order, with root node at node_id 0.
    :param Tensor parent: Maps node_id of child nodes to node_id of parent,
        or to -1 for the single root node (which must have node_id 0).
    :param Tensor state_init: Maps node_id to initial/observed distribution.
        Entries are typically zeros for internal nodes and log-one-hot vectors
        for leaves, but may be arbitrary probability vectors.
    :param Tensor state_trans: Either a homogeneous reverse-time state
        transition matrix, or a heterogeneous grid of ``T`` transition matrices
        applying to time intervals ``(-inf,1]``, ``(1,2]``, ..., ``(T-1,inf)``.
    :param Tensor coal_rate: Either a homogeneous coalescent rate or a
        heterogeneous grid of ``T`` rates applying to time intervals
        ``(-inf,1]``, ``(1,2]``, ..., ``(T-1,inf)``.
    """
    num_nodes, = time.shape
    assert parent.shape == (num_nodes,)
    num_states = state_init.size(-1)
    assert state_trans.shape[-2:] == (num_states, num_states)
    assert coal_rate.size(-1) == num_states
    if validate_args:
        constraints.simplex.check(state_init)
        constraints.simplex.check(state_trans)

    log_prob = state_init.clone()

    num_pending = torch.zeros(num_nodes)
    for i in range(num_nodes):
        num_pending[parent[i]] += 1

    for i in range(-1 - num_nodes, 0):
        j = parent[i]
        # State transition part.
        log_prob[j] += _interpolate_mv(time[i], time[j], state_trans, log_prob[i])
        # Survival probability aka interval part.
        log_prob[j] -= _integrate(time[i], time[j], coal_rate)
        # Binomial coalescent event part.
        log_prob[j] += _interpolate(time[j], coal_rate)
        # Probablity that both children are in the same state.
        # This assumes binary coalescence.
        num_pending[j] -= 1
        if num_pending[j] == 0:
            log_prob[j] += log_prob[j] - log_prob[j].logsumexp(-1)
    return log_prob[0].logsumexp(-1)
