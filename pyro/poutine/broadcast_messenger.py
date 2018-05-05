from __future__ import absolute_import, division, print_function

from .messenger import Messenger


class BroadcastMessenger(Messenger):
    """
    `BroadcastMessenger` automatically broadcasts the batch shape of
    the stochastic function at a sample site when inside a single
    or nested iarange context. The existing `batch_shape` must be
    broadcastable with the size of the :class::`pyro.iarange`
    contexts installed in the `cond_indep_stack`.
    """
    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        """
        if msg["done"] or msg["type"] != "sample":
            return

        dist = msg["fn"]
        actual_batch_shape = getattr(dist, "batch_shape", None)
        if actual_batch_shape is not None:
            target_batch_shape = []
            for f in msg["cond_indep_stack"]:
                if f.dim is not None:
                    assert f.dim < 0
                    target_batch_shape = [None] * (-f.dim - len(target_batch_shape)) + target_batch_shape
                    if target_batch_shape[f.dim] is not None:
                        raise ValueError('\n  '.join([
                            'at site "{}" within iarange("", dim={}), dim collision'
                            .format(msg["name"], f.name, f.dim),
                            'Try setting dim arg in other iaranges.']))
                    target_batch_shape[f.dim] = f.size
            # If expected shape is None at an index, infer as either 1,
            # or the actual shape starting from the right.
            for i in range(-len(target_batch_shape)+1, 1):
                if target_batch_shape[i] is None:
                    target_batch_shape[i] = actual_batch_shape[i] if len(actual_batch_shape) > -i else 1
            msg["fn"] = msg["fn"].expand(target_batch_shape)
