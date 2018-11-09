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
        self._enum_dims = {}  # site name -> reusable dim (negative integer)
        self._enum_symbols = {}  # site name -> unique symbol (nonnegative integer)
        self._markov_depths = {}  # site name -> depth (nonnegative integer)

        self._param_dim_to_symbol = {}  # site name -> (enum dim -> unique symbol)
        self._value_dim_to_symbol = {}  # site name -> (enum dim -> unique symbol)
        return super(EnumerateMessenger, self).__enter__()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"]:
            return

        upstream = msg["infer"].get("_markov_upstream")  # site name -> markov depth
        dim_to_symbol = {}  # reusable dim -> unique symbol
        if upstream is not None:
            upstream_dims = set()
            for name, depth in upstream.items():
                if self._markov_depths[name] == depth:  # hide sites whose markov context has exited
                    dim = self._enum_dims[name]
                    upstream_dims.add(dim)
                    dim_to_symbol[dim] = self._enum_symbols[name]
            self._markov_depths[msg["name"]] = msg["infer"]["_markov_depth"]
        msg["infer"]["_dim_to_symbol"] = dim_to_symbol
        if msg["is_observed"] or msg["infer"].get("enumerate") != "parallel":
            return

        dist = msg["fn"]
        num_samples = msg["infer"].get("num_samples")
        if num_samples is None:
            # Enumerate over the support of the distribution.
            value = dist.enumerate_support(expand=msg["infer"].get("expand", False))
        else:
            # Monte Carlo sample the distribution.
            value = dist(sample_shape=(num_samples,))
        event_dim = len(dist.event_shape)
        assert value.dim() == 1 + len(dist.batch_shape) + event_dim

        # Ensure enumeration happens at an available tensor dimension.
        # This allocates the next available dim for enumeration, to the left all other dims.
        actual_dim = -1 - len(dist.batch_shape)  # the leftmost dim of log_prob, indexed from the right

        # Find a target_dim, possibly different from actual_dim.
        target_dim, symbol = _ENUM_ALLOCATOR.allocate()
        if upstream is not None:
            self._enum_dims[msg["name"]] = target_dim
            self._enum_symbols[msg["name"]] = symbol
        dim_to_symbol[target_dim] = symbol

        # Reshape to move actual_dim to target_dim.
        if actual_dim < target_dim:
            assert value.size(target_dim - event_dim) == 1, 'pyro.markov dim conflict at dim {}'.format(actual_dim)
            value = value.transpose(target_dim - event_dim, actual_dim - event_dim)
            while value.dim() and value.size(0) == 1:
                value = value.squeeze(0)
        elif target_dim < actual_dim:
            diff = actual_dim - target_dim
            value = value.reshape(value.shape[:1] + (1,) * diff + value.shape[1:])

        shape = msg["fn"].batch_shape
        self._param_dim_to_symbol = {d: s for d, s in dim_to_symbol.items() if len(shape) > -d and shape[d] > 1}
        shape = value.shape[:value.dim() - event_dim]
        self._value_dim_to_symbol = {d: s for d, s in dim_to_symbol.items() if len(shape) > -d and shape[d] > 1}
        print('DEBUG {} dim = {}'.format(msg["name"], target_dim))

        msg["infer"]["_enumerate_dim"] = target_dim
        msg["value"] = value
        msg["done"] = True
