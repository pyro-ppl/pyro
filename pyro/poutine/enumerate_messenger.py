from __future__ import absolute_import, division, print_function

from .messenger import Messenger
from .runtime import _ENUM_ALLOCATOR


class EnumerateMessenger(Messenger):
    """
    Enumerates in parallel over discrete sample sites marked
    ``infer={"enumerate": "parallel"}``.

    :param first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This can be an integer or a callable returning an integer.
    :type first_available_dim: int or callable
    """
    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        super(EnumerateMessenger, self).__init__()

    def __enter__(self):
        if self.first_available_dim is not None:
            _ENUM_ALLOCATOR.set_first_available_dim(self.first_available_dim)
        self._enum_dims = {}
        self._enum_symbols = {}
        self._markov_depths = {}
        return super(EnumerateMessenger, self).__enter__()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"] or msg["type"] != "sample" or msg["is_observed"]:
            return
        if msg["infer"].get("enumerate") != "parallel":
            return

        dist = msg["fn"]
        num_samples = msg["infer"].get("num_samples")
        if num_samples is None:
            # Enumerate over the support of the distribution.
            value = dist.enumerate_support(expand=msg["infer"].get("expand", False))
        else:
            # Monte Carlo sample the distribution.
            value = dist(sample_shape=(num_samples,))
        assert len(value.shape) == 1 + len(dist.batch_shape) + len(dist.event_shape)

        # Ensure enumeration happens at an available tensor dimension.
        # This allocates the next available dim for enumeration, to the left all other dims.
        actual_dim = -1 - len(dist.batch_shape)  # the leftmost dim of log_prob, counting from the right

        # Find a target_dim, possibly even farther left than actual_dim.
        upstream = msg["infer"].get("_markov_upstream")
        if upstream is None:
            target_dim, symbol = _ENUM_ALLOCATOR.allocate()
            msg["infer"]["_dim_to_symbol"] = {target_dim: symbol}
        else:
            upstream_names = set(name for name, depth in upstream.items()
                                 if self._markov_depths[name] == depth)
            upstream_dims = set(map(self._enum_dims.__getitem__, upstream_names))
            target_dim, symbol = _ENUM_ALLOCATOR.allocate(upstream_dims)
            self._enum_dims[msg["name"]] = target_dim
            self._enum_symbols[msg["name"]] = symbol
            self._markov_depths[msg["name"]] = msg["infer"]["_markov_depth"]
            msg["infer"]["_dim_to_symbol"] = {self._enum_dims[name]: self._enum_symbols[name]
                                              for name in upstream_names}

        # Reshape to move actual_dim to target_dim.
        if target_dim < actual_dim:
            assert value.size(actual_dim) == 1, 'pyro.markov dim conflict at dim {}'.format(actual_dim)
            value = value.transpose(target_dim, actual_dim)
            while value.size(0) == 1:
                value = value.squeeze(0)
        elif target_dim > actual_dim:
            diff = target_dim - actual_dim
            value = value.reshape(value.shape[:1] + (1,) * diff + value.shape[1:])

        msg["infer"]["_enumerate_dim"] = -1 - target_dim
        msg["value"] = value
        msg["done"] = True
