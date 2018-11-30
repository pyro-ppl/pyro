from __future__ import absolute_import, division, print_function

from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.util import ignore_jit_warnings

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

    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
        This should be a negative integer or None.
    """
    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        super(EnumerateMessenger, self).__init__()

    def __enter__(self):
        if self.first_available_dim is not None:
            _ENUM_ALLOCATOR.set_first_available_dim(self.first_available_dim)
        self._markov_depths = {}  # site name -> depth (nonnegative integer)
        self._param_dims = {}  # site name -> (enum dim -> unique id)
        self._value_dims = {}  # site name -> (enum dim -> unique id)
        return super(EnumerateMessenger, self).__enter__()

    @ignore_jit_warnings()
    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"] or not isinstance(msg["fn"], TorchDistributionMixin):
            return

        # Compute upstream dims in scope; these are unsafe to use for this site's target_dim.
        scope = msg["infer"].get("_markov_scope")  # site name -> markov depth
        param_dims = _ENUM_ALLOCATOR.dim_to_id.copy()  # enum dim -> unique id
        if scope is not None:
            for name, depth in scope.items():
                if self._markov_depths[name] == depth:  # hide sites whose markov context has exited
                    param_dims.update(self._value_dims[name])
            self._markov_depths[msg["name"]] = msg["infer"]["_markov_depth"]
        self._param_dims[msg["name"]] = param_dims
        if msg["is_observed"] or msg["infer"].get("enumerate") != "parallel":
            return

        # Compute an enumerated value (at an arbitrary dim).
        value = enumerate_site(msg)
        actual_dim = -1 - len(msg["fn"].batch_shape)  # the leftmost dim of log_prob

        # Move actual_dim to a safe target_dim.
        target_dim, id_ = _ENUM_ALLOCATOR.allocate(None if scope is None else param_dims)
        event_dim = msg["fn"].event_dim
        if actual_dim < target_dim:
            assert value.size(target_dim - event_dim) == 1, \
                'pyro.markov dim conflict at dim {}'.format(actual_dim)
            value = value.transpose(target_dim - event_dim, actual_dim - event_dim)
            while value.dim() and value.size(0) == 1:
                value = value.squeeze(0)
        elif target_dim < actual_dim:
            diff = actual_dim - target_dim
            value = value.reshape(value.shape[:1] + (1,) * diff + value.shape[1:])

        # Compute dims passed downstream through the value.
        value_dims = {dim: param_dims[dim] for dim in range(event_dim - value.dim(), 0)
                      if value.size(dim - event_dim) > 1 and dim in param_dims}
        value_dims[target_dim] = id_

        msg["infer"]["_enumerate_dim"] = target_dim
        msg["infer"]["_dim_to_id"] = value_dims
        msg["value"] = value
        msg["done"] = True

    def _pyro_post_sample(self, msg):
        # Save all dims exposed in this sample value.
        # Whereas all of site["_dim_to_id"] are needed to interpret a
        # site's log_prob tensor, only a filtered subset self._value_dims[msg["name"]]
        # are needed to interpret a site's value.
        if not isinstance(msg["fn"], TorchDistributionMixin):
            return
        value = msg["value"]
        if value is None:
            return
        shape = value.shape[:value.dim() - msg["fn"].event_dim]
        dim_to_id = msg["infer"].setdefault("_dim_to_id", {})
        dim_to_id.update(self._param_dims.get(msg["name"], {}))
        with ignore_jit_warnings():
            self._value_dims[msg["name"]] = {dim: id_ for dim, id_ in dim_to_id.items()
                                             if len(shape) >= -dim and shape[dim] > 1}
