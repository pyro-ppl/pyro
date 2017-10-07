class NonlocalExit(Exception):
    """
    TODO doc
    nonlocal exit
    """
    def __init__(self, site, *args, **kwargs):
        """
        TODO doc
        constructor
        """
        super(NonlocalExit, self).__init__(*args, **kwargs)
        self.site = site


def enum_extend(trace, msg, num_samples=None):
    """
    TODO doc
    """
    if num_samples is None:
        num_samples = -1

    extended_traces = []
    for i, s in enumerate(msg["fn"].support(*msg["args"], **msg["kwargs"])):
        if i > num_samples and num_samples >= 0:
            break
        msg_copy = msg.copy()
        msg_copy.update({"ret": s})
        extended_traces.append(trace.copy().add_sample(
            msg_copy["name"], msg_copy["scale"], msg_copy["ret"],
            msg_copy["fn"], *msg_copy["args"], **msg_copy["kwargs"]))
    return extended_traces


def mc_extend(trace, msg, num_samples=None):
    """
    TODO doc
    """
    if num_samples is None:
        num_samples = 1

    extended_traces = []
    for i in range(num_samples):
        msg_copy = msg.copy()
        msg_copy.update({"ret": msg["fn"](*msg["args"], **msg["kwargs"])})
        extended_traces.append(trace.copy().add_sample(
            msg_copy["name"], msg_copy["scale"], msg_copy["ret"],
            msg_copy["fn"], *msg_copy["args"], **msg_copy["kwargs"]))
    return extended_traces


def discrete_escape(trace, msg):
    """
    TODO doc
    """
    return (msg["type"] == "sample") and \
        (msg["name"] not in trace) and \
        (hasattr(msg["fn"], "enumerable")) and \
        (msg["fn"].enumerable)


def all_escape(trace, msg):
    """
    TODO doc
    """
    return (msg["type"] == "sample") and \
        (msg["name"] not in trace)
