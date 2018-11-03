from __future__ import absolute_import, division, print_function

import itertools

from .markov_messenger import MarkovMessenger
from .messenger import Messenger


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
    def __init__(self, first_available_dim, first_available_symbol=0):
        super(EnumerateMessenger, self).__init__()
        self.first_available_dim = first_available_dim
        self.first_available_symbol = first_available_symbol
        self.next_available_dim = None
        self.next_available_symbol = None

    def __call__(self, fn):
        fn = MarkovMessenger()(fn)
        return super(EnumerateMessenger, self).__call__(fn)

    def __enter__(self):
        first = self.first_available_dim
        self._first_available_dim = first() if callable(first) else first
        self.next_available_dim = self._first_available_dim
        first = self.first_available_symbol
        self.next_available_symbol = first() if callable(first) else first
        return super(EnumerateMessenger, self).__enter__()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"] or msg["type"] != "sample" or msg["is_observed"]:
            return

        if msg["infer"].get("enumerate") == "parallel":
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
            actual_dim = len(dist.batch_shape)  # the leftmost dim of log_prob, counting from the right

            # Find a target_dim, possibly even farther left than actual_dim.
            for target_dim in itertools.count(self._first_available_dim):
                if target_dim not in msg["cond_dep_set"]:
                    break
            self.next_available_dim = max(self.next_available_dim, 1 + target_dim)
            symbol = self.next_available_symbol
            self.next_available_symbol += 1

            if target_dim == float('inf'):
                raise ValueError("max_plate_nesting must be set to a finite value for parallel enumeration")
            if actual_dim > target_dim:
                raise ValueError("Expected enumerated value to have dim at most {} but got shape {}".format(
                    target_dim + len(dist.event_shape), value.shape))
            elif target_dim > actual_dim:
                # Reshape to move actual_dim to target_dim.
                diff = target_dim - actual_dim
                value = value.reshape(value.shape[:1] + (1,) * diff + value.shape[1:])

            msg["infer"]["_enumerate_dim"] = -1 - target_dim
            msg["infer"]["_enumerate_symbol"] = symbol
            msg["value"] = value
            msg["done"] = True
