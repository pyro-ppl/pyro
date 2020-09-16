# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints


def _mv(matrix, log_vector):
    shift = log_vector.logsumexp(-1)
    p = (log_vector - shift)
    q = matrix.matmul(p.unsqueeze(-1)).squeeze(-1)
    return q + shift


def _interpolate_mv(t0, t1, matrix, log_vector):
    if matrix.dim() == 2 or matrix.size(-3) == 1:
        # homogeneous
        m = matrix.exp().matrix_power(t1 - t0)
        return _mv(matrix, log_vector)
    raise NotImplementedError("TODO support time-inhomogeneous transitions")


def markov_tree_log_prob(
    time, parent, leaves, leaf_state, state_trans, *,
    validate_args=None,
):
    """
    :param Tensor time: Maps node_id to the time of that node.
        This should be sorted in ascending order, with root node at node_id 0.
    :param Tensor parent: Maps node_id of child nodes to node_id of parent,
        or to -1 for the single root node (which must have node_id 0).
    :param Tensor leaves: int tensor of ids of all leaf nodes.
    :param Tensor leaf_state: int tensor of states of all leaf nodes.
    :param Tensor state_trans: Either a homogeneous reverse-time state
        transition matrix, or a heterogeneous grid of ``T`` transition matrices
        applying to time intervals ``(-inf,1]``, ``(1,2]``, ..., ``(T-1,inf)``.
    """
    assert time.dim() == 1
    num_nodes, = time.shape
    num_states = state_trans.size(-1)
    num_leaves = (num_nodes + 1) // 2
    assert leaves.shape == time.shape[:-1] + (num_leaves,)
    assert time.dim() == 1
    assert time.shape == (num_nodes,)
    assert parent.shape == (num_nodes,)
    assert state_trans.dim() in (2, 3)  # homogeneous, heterogeneous
    assert state_trans.shape[-2:] == (num_states, num_states)

    # Convert (leaves,leaf_state) -> state_init.
    state_init = state_trans.new_zeros(num_nodes, num_states)
    state_init[leaves] = -math.inf
    state_init[leaves, leaf_state] = 0

    return _markov_tree_log_prob(time, parent, state_init, state_trans,
                                 validate_args=validate_args)


def _markov_tree_log_prob(
    time, parent, state_init, state_trans, *,
    validate_args=None,
):
    """
    **References:**

    [1] T. Vaughan, D. Kuhnert, A. Popinga, D. Welch, A. Drummond (2014)
        `Efficient Bayesian inference under the structured coalescent`
        https://academic.oup.com/bioinformatics/article/30/16/2272/2748160
    [2] Peter Beerli, Joseph Felsenstein (2001)
        `Maximum likelihood estimation of a migration matrix and effective
        population sizes in n subpopulations by using a coalescent approach`
        https://www.pnas.org/content/98/8/4563

    :param Tensor time: Maps node_id to the time of that node.
        This should be sorted in ascending order, with root node at node_id 0.
    :param Tensor parent: Maps node_id of child nodes to node_id of parent,
        or to -1 for the single root node (which must have node_id 0).
    :param Tensor state_init: Maps node_id to initial/observed Categorical
        distribution by logits. Entries are typically zeros for internal nodes
        and log-one-hot vectors for leaves, but may be arbitrary
        log-probability vectors.
    :param Tensor state_trans: Either a homogeneous reverse-time state
        transition matrix, or a heterogeneous grid of ``T`` transition matrices
        applying to time intervals ``(-inf,1]``, ``(1,2]``, ..., ``(T-1,inf)``.
    """
    assert time.dim() == 1
    num_nodes, num_states = state_init.shape
    assert time.shape == (num_nodes,)
    assert parent.shape == (num_nodes,)
    assert state_trans.dim() in (2, 3)  # homogeneous, heterogeneous
    assert state_trans.shape[-2:] == (num_states, num_states)
    if validate_args:
        assert (time[:-1] <= time[1:]).all(), "time is not ascending"
        assert parent[0] == -1
        assert parent[1:] != -1
        constraints.simplex.check(state_init.exp())  # FIXME only holds for leaves
        constraints.simplex.check(state_trans.exp())

    log_prob = state_init.clone()
    for i in range(-num_nodes, 0, -1):
        j = parent[i]
        log_prob[j] += _interpolate_mv(time[i], time[j], state_trans, log_prob[i])
    return log_prob[0].logsumexp(-1)
