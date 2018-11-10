from __future__ import absolute_import, division, print_function

from pyro.distributions.torch_distribution import TorchDistributionMixin

from .messenger import Messenger
from .runtime import _ENUM_ALLOCATOR


def enumerate_site(msg):
    dist = msg["fn"]
    num_samples = msg["infer"].get("num_samples")
    if num_samples is None:
        # Enumerate over the support of the distribution.
        value = dist.enumerate_support(expand=msg["infer"].get("expand", False))
    else:
        # Monte Carlo sample the distribution.
        value = dist(sample_shape=(num_samples,))
    assert value.dim() == 1 + len(dist.batch_shape) + len(dist.event_shape)
    return value


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
        self._markov_depths = {}  # site name -> depth (nonnegative integer)
        self._dim_to_symbol = {}  # site name -> (enum dim -> unique symbol)
        return super(EnumerateMessenger, self).__enter__()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"] or not isinstance(msg["fn"], TorchDistributionMixin):
            return

        # Compute dims in scope; these are unsafe to use for this site's target_dim.
        scope = msg["infer"].get("_markov_scope")  # site name -> markov depth
        scope_dims = None
        dim_to_symbol = {}  # reusable dim -> unique symbol
        if scope is not None:
            scope_dims = set()
            for name, depth in scope.items():
                if self._markov_depths[name] == depth:  # hide sites whose markov context has exited
                    dim_to_symbol.update(self._dim_to_symbol[name])
                    scope_dims.update(self._dim_to_symbol[name])
            self._markov_depths[msg["name"]] = msg["infer"]["_markov_depth"]
        msg["infer"]["_dim_to_symbol"] = dim_to_symbol
        if msg["is_observed"] or msg["infer"].get("enumerate") != "parallel":
            return

        # Compute an enumerated value (at an arbitrary dim).
        value = enumerate_site(msg)
        actual_dim = -1 - len(msg["fn"].batch_shape)  # the leftmost dim of log_prob

        # Move actual_dim to a safe target_dim.
        target_dim, symbol = _ENUM_ALLOCATOR.allocate(scope_dims)
        if actual_dim < target_dim:
            event_dim = msg["fn"].event_dim
            assert value.size(target_dim - event_dim) == 1, \
                'pyro.markov dim conflict at dim {}'.format(actual_dim)
            value = value.transpose(target_dim - event_dim, actual_dim - event_dim)
            while value.dim() and value.size(0) == 1:
                value = value.squeeze(0)
        elif target_dim < actual_dim:
            diff = actual_dim - target_dim
            value = value.reshape(value.shape[:1] + (1,) * diff + value.shape[1:])

        dim_to_symbol[target_dim] = symbol
        msg["infer"]["_enumerate_dim"] = target_dim
        msg["value"] = value
        msg["done"] = True

    def _pyro_post_sample(self, msg):
        # Save all dims exposed in this sample value.
        # Whereas dim_to_symbol is needed to interpret a site's log_prob tensor,
        # only a filtered subset is needed to interpret a sites value.
        value = msg["value"]
        dim_to_symbol = msg["infer"].get("_dim_to_symbol")
        if value is None or dim_to_symbol is None:
            return
        shape = value.shape[:value.dim() - msg["fn"].event_dim]
        self._dim_to_symbol[msg["name"]] = {dim: symbol
                                            for dim, symbol in dim_to_symbol.items()
                                            if len(shape) >= -dim and shape[dim] > 1}
