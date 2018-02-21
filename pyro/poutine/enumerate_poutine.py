from __future__ import absolute_import, division, print_function

from .poutine import Messenger, Poutine


def _iter_discrete_filter(name, msg):
    return ((msg["type"] == "sample") and
            (not msg["is_observed"]) and
            getattr(msg["fn"], "enumerable", False) and
            (msg["infer"].get("enumerate", "sequential") == "parallel"))


class EnumerateMessenger(Messenger):
    """
    Enumerates in parallel over discrete sample sites marked
    ``infer={"enumerate": "parallel"}``.

    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
    """
    def __init__(self, first_available_dim=0):
        super(EnumerateMessenger, self).__init__()
        self.next_available_dim = first_available_dim

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if not msg["done"] and _iter_discrete_filter(msg["name"], msg):
            value = msg["fn"].enumerate_support()

            # Ensure enumeration happens at an available tensor dimension.
            event_dim = len(msg["fn"].event_shape)
            actual_dim = value.dim() - event_dim - 1
            target_dim = self.next_available_dim
            self.next_available_dim += 1
            if actual_dim > target_dim:
                raise ValueError("Expected enumerated value to have dim at most {} but got shape {}".format(
                    target_dim + event_dim, value.shape))
            elif target_dim > actual_dim:
                diff = target_dim - actual_dim
                value = value.contiguous().view(value.shape[:1] + (1,) * diff + value.shape[1:])

            msg["value"] = value
            msg["done"] = True


class EnumeratePoutine(Poutine):
    """
    Enumerates in parallel over discrete sample sites that are configured with
    ``infer={"enumerate": "parallel"}``.

    :param int first_available_dim: The first tensor dimension (counting
        from the right) that is available for parallel enumeration. This
        dimension and all dimensions left may be used internally by Pyro.
    """
    def __init__(self, fn, first_available_dim=0):
        """
        :param fn: A stochastic function (callable containing pyro primitive
            calls).
        :param int first_available_dim: The first tensor dimension (counting
            from the right) that is available for parallel enumeration. This
            dimension and all dimensions left may be used internally by Pyro.
        """
        super(EnumeratePoutine, self).__init__(EnumerateMessenger(first_available_dim), fn)
