import torch


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
    :param Tensor time: Maps node_id to the time of that node.
        This should be sorted in ascending order, with root node at node_id 0.
    :param Tensor parent: Maps node_id of child nodes to node_id of parent,
        or to -1 for the single root node (which must have node_id 0).
    :param Tensor state_init: Maps node_id to initial/observed distribution.
        Entries are typically zeros for internal nodes and log-one-hot vectors
        for leaves, but may be arbitrary probability vectors.
    :param Tensor state_trans: Either a homogeneous reverse-time state
        transition matrix, or a heterogeneous grid of ``T`` transition matrices
        applying to time intervals ``(-inf,1]``, ``(1, 2]``, ..., ``(T-1,inf)``.
    :param Tensor coal_rate: Either a homogeneous coalescent rate or a
        heterogeneous grid of ``T`` rates applying to time intervals
        ``(-inf,1]``, ``(1, 2]``, ..., ``(T-1,inf)``.
    """
    num_nodes, = time.shape
    assert parent.shape == (num_nodes,)
    if validate_args:
        constraints.simplex.check(state_init)
        constraints.simplex.check(state_trans)

    log_prob = state_init.clone()
    for i in range(-1 - num_nodes, 0):
        j = parent[i]
        log_prob[j] += _interpolate_mv(time[i], time[j], state_trans, log_prob[i])
    return log_prob[0].logsumexp(-1)
