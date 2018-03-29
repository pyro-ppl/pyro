from __future__ import absolute_import, division, print_function

from .poutine import _PYRO_STACK


def site_is_subsample(site):
    """
    Determines whether a trace site originated from a subsample statement inside an `iarange`.
    """
    return site["type"] == "sample" and type(site["fn"]).__name__ == "_Subsample"


def prune_subsample_sites(trace):
    """
    Copies and removes all subsample sites from a trace.
    """
    trace = trace.copy()
    for name, site in list(trace.nodes.items()):
        if site_is_subsample(site):
            trace.remove_node(name)
    return trace


class NonlocalExit(Exception):
    """
    Exception for exiting nonlocally from poutine execution.

    Used by poutine.EscapePoutine to return site information.
    """
    def __init__(self, site, *args, **kwargs):
        """
        :param site: message at a pyro site

        constructor.  Just stores the input site.
        """
        super(NonlocalExit, self).__init__(*args, **kwargs)
        self.site = site

    def reset_stack(self):
        """
        Reset the state of the frames remaining in the stack.
        Necessary for multiple re-executions in poutine.queue.
        """
        for frame in _PYRO_STACK:
            frame._reset()


def enum_extend(trace, msg, num_samples=None):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :param num_samples: maximum number of extended traces to return.
    :returns: a list of traces, copies of input trace with one extra site

    Utility function to copy and extend a trace with sites based on the input site
    whose values are enumerated from the support of the input site's distribution.

    Used for exact inference and integrating out discrete variables.
    """
    if num_samples is None:
        num_samples = -1

    extended_traces = []
    for i, s in enumerate(msg["fn"].enumerate_support(*msg["args"], **msg["kwargs"])):
        if i > num_samples and num_samples >= 0:
            break
        msg_copy = msg.copy()
        msg_copy.update(value=s)
        tr_cp = trace.copy()
        tr_cp.add_node(msg["name"], **msg_copy)
        extended_traces.append(tr_cp)
    return extended_traces


def mc_extend(trace, msg, num_samples=None):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :param num_samples: maximum number of extended traces to return.
    :returns: a list of traces, copies of input trace with one extra site

    Utility function to copy and extend a trace with sites based on the input site
    whose values are sampled from the input site's function.

    Used for Monte Carlo marginalization of individual sample sites.
    """
    if num_samples is None:
        num_samples = 1

    extended_traces = []
    for i in range(num_samples):
        msg_copy = msg.copy()
        msg_copy["value"] = msg_copy["fn"](*msg_copy["args"], **msg_copy["kwargs"])
        tr_cp = trace.copy()
        tr_cp.add_node(msg_copy["name"], **msg_copy)
        extended_traces.append(tr_cp)
    return extended_traces


def discrete_escape(trace, msg):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :returns: boolean decision value

    Utility function that checks if a sample site is discrete and not already in a trace.

    Used by EscapePoutine to decide whether to do a nonlocal exit at a site.
    Subroutine for integrating out discrete variables for variance reduction.
    """
    return (msg["type"] == "sample") and \
        (not msg["is_observed"]) and \
        (msg["name"] not in trace) and \
        (getattr(msg["fn"], "has_enumerate_support", False))


def all_escape(trace, msg):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :returns: boolean decision value

    Utility function that checks if a site is not already in a trace.

    Used by EscapePoutine to decide whether to do a nonlocal exit at a site.
    Subroutine for approximately integrating out variables for variance reduction.
    """
    return (msg["type"] == "sample") and \
        (not msg["is_observed"]) and \
        (msg["name"] not in trace)


def broadcast_enum_filter(msg):
    if msg["done"] or msg["type"] != "sample" or msg["is_observed"]:
        return False
    return msg["infer"].get("enumerate") == "parallel"


def broadcast_enum_cont(msg):
    """
    :param msg: current message at a trace site.
    """
    # Enumerate over the support of the distribution.
    dist = msg["fn"]
    value = dist.enumerate_support()
    assert len(value.shape) == 1 + len(dist.batch_shape) + len(dist.event_shape)

    # Ensure enumeration happens at an available tensor dimension.
    # This allocates the next available dim for enumeration, to the left all other dims.
    actual_dim = len(dist.batch_shape)  # the leftmost dim of log_prob, counting from the right
    target_dim = msg["infer"]["next_available_dim"]  # possibly even farther left than actual_dim
    msg["infer"]["next_available_dim"] += 1
    if target_dim == float('inf'):
        raise ValueError("max_iarange_nesting must be set to a finite value for parallel enumeration")
    if actual_dim > target_dim:
        raise ValueError("Expected enumerated value to have dim at most {} but got shape {}".format(
            target_dim + len(dist.event_shape), value.shape))
    elif target_dim > actual_dim:
        # Reshape to move actual_dim to target_dim.
        diff = target_dim - actual_dim
        value = value.contiguous().view(value.shape[:1] + (1,) * diff + value.shape[1:])

    msg["value"] = value


def escape_cont_fn(msg):
    raise NonlocalExit(msg)
