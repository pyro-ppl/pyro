import sys

from .messenger import Messenger
from .trace_struct import Trace
from .util import site_is_subsample


def identify_dense_edges(trace):
    """
    Modifies a trace in-place by adding all edges based on the
    `cond_indep_stack` information stored at each site.
    """
    for name, node in trace.nodes.items():
        if site_is_subsample(node):
            continue
        if node["type"] == "sample":
            for past_name, past_node in trace.nodes.items():
                if site_is_subsample(node):
                    continue
                if past_node["type"] == "sample":
                    if past_name == name:
                        break
                    past_node_independent = False
                    for query, target in zip(node["cond_indep_stack"], past_node["cond_indep_stack"]):
                        if query.name == target.name and query.counter != target.counter:
                            past_node_independent = True
                            break
                    if not past_node_independent:
                        trace.add_edge(past_name, name)


class TraceMessenger(Messenger):
    """
    Execution trace messenger.

    A TraceMessenger records the input and output to every Pyro primitive
    and stores them as a site in a Trace().
    This should, in theory, be sufficient information for every inference algorithm
    (along with the implicit computational graph in the Variables?)

    We can also use this for visualization.
    """

    def __init__(self, graph_type=None, param_only=None):
        """
        :param string graph_type: string that specifies the type of graph
            to construct (currently only "flat" or "dense" supported)
        :param param_only: boolean that specifies whether to record sample sites
        """
        super(TraceMessenger, self).__init__()
        if graph_type is None:
            graph_type = "flat"
        if param_only is None:
            param_only = False
        assert graph_type in ("flat", "dense")
        self.graph_type = graph_type
        self.param_only = param_only
        self.trace = Trace(graph_type=self.graph_type)

    def __enter__(self):
        self.trace = Trace(graph_type=self.graph_type)
        return super(TraceMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        """
        Adds appropriate edges based on cond_indep_stack information
        upon exiting the context.
        """
        if self.param_only:
            for node in list(self.trace.nodes.values()):
                if node["type"] != "param":
                    self.trace.remove_node(node["name"])
        if self.graph_type == "dense":
            identify_dense_edges(self.trace)
        return super(TraceMessenger, self).__exit__(*args, **kwargs)

    def __call__(self, fn):
        """
        TODO docs
        """
        return TraceHandler(self, fn)

    def get_trace(self):
        """
        :returns: data structure
        :rtype: pyro.poutine.Trace

        Helper method for a very common use case.
        Returns a shallow copy of ``self.trace``.
        """
        return self.trace.copy()

    def _reset(self):
        tr = Trace(graph_type=self.graph_type)
        if "_INPUT" in self.trace.nodes:
            tr.add_node("_INPUT",
                        name="_INPUT", type="input",
                        args=self.trace.nodes["_INPUT"]["args"],
                        kwargs=self.trace.nodes["_INPUT"]["kwargs"])
        self.trace = tr
        super(TraceMessenger, self)._reset()

    def _pyro_post_sample(self, msg):
        if not self.param_only:
            # RWS: could change msg here
            self.trace.add_node(msg["name"], **msg.copy())

    def _pyro_post_param(self, msg):
        self.trace.add_node(msg["name"], **msg.copy())


class TraceHandler(object):
    """
    Execution trace poutine.

    A TraceHandler records the input and output to every Pyro primitive
    and stores them as a site in a Trace().
    This should, in theory, be sufficient information for every inference algorithm
    (along with the implicit computational graph in the Variables?)

    We can also use this for visualization.
    """
    def __init__(self, msngr, fn):
        self.fn = fn
        self.msngr = msngr

    def __call__(self, *args, **kwargs):
        """
        Runs the stochastic function stored in this poutine,
        with additional side effects.

        Resets self.trace to an empty trace,
        installs itself on the global execution stack,
        runs self.fn with the given arguments,
        uninstalls itself from the global execution stack,
        stores the arguments and return value of the function in special sites,
        and returns self.fn's return value
        """
        with self.msngr:
            self.msngr.trace.add_node("_INPUT",
                                      name="_INPUT", type="args",
                                      args=args, kwargs=kwargs)
            try:
                # RWS: fn is actually the guide/model
                ret = self.fn(*args, **kwargs)
            except (ValueError, RuntimeError):
                exc_type, exc_value, traceback = sys.exc_info()
                shapes = self.msngr.trace.format_shapes()
                raise exc_type(u"{}\n{}".format(exc_value, shapes)).with_traceback(traceback)
            self.msngr.trace.add_node("_RETURN", name="_RETURN", type="return", value=ret)
        return ret

    @property
    def trace(self):
        return self.msngr.trace

    def get_trace(self, *args, **kwargs):
        """
        :returns: data structure
        :rtype: pyro.poutine.Trace

        Helper method for a very common use case.
        Calls this poutine and returns its trace instead of the function's return value.
        """
        self(*args, **kwargs)
        return self.msngr.get_trace()
