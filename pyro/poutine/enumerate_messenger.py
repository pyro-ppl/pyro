from __future__ import absolute_import, division, print_function

from .messenger import Messenger


class EnumerateMessenger(Messenger):
    """
    Enumerates in parallel over discrete sample sites marked
    ``infer={"enumerate": "parallel"}``.

    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
    """
    def __init__(self, first_available_dim):
        super(EnumerateMessenger, self).__init__()
        self.first_available_dim = first_available_dim
        self.next_available_dim = None

    def __enter__(self):
        self.next_available_dim = self.first_available_dim
        return super(EnumerateMessenger, self).__enter__()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"] or msg["type"] != "sample" or msg["is_observed"]:
            return

        if msg["infer"].get("enumerate") == "parallel":
            # Enumerate over the support of the distribution.
            dist = msg["fn"]
            value = dist.enumerate_support()
            assert len(value.shape) == 1 + len(dist.batch_shape) + len(dist.event_shape)

            # Ensure enumeration happens at an available tensor dimension.
            # This allocates the next available dim for enumeration, to the left all other dims.
            actual_dim = len(dist.batch_shape)  # the leftmost dim of log_prob, counting from the right
            target_dim = self.next_available_dim  # possibly even farther left than actual_dim
            self.next_available_dim += 1
            if target_dim == float('inf'):
                raise ValueError("max_iarange_nesting must be set to a finite value for parallel enumeration")
            if actual_dim > target_dim:
                raise ValueError("Expected enumerated value to have dim at most {} but got shape {}".format(
                    target_dim + len(dist.event_shape), value.shape))
            elif target_dim > actual_dim:
                # Reshape to move actual_dim to target_dim.
                diff = target_dim - actual_dim
                value = value.reshape(value.shape[:1] + (1,) * diff + value.shape[1:])

            msg["value"] = value
            msg["done"] = True
