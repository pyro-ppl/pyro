import collections
import networkx


class Trace(networkx.DiGraph):
    """
    Execution trace data structure
    """

    node_dict_factory = collections.OrderedDict

    def __init__(self, *args, **kwargs):
        """
        :param string graph_type: string specifying the kind of trace graph to construct

        Constructor. Currently identical to networkx.DiGraph(*args, **kwargs),
        except for storing the graph_type attribute
        """
        graph_type = kwargs.pop("graph_type", "flat")
        assert graph_type in ("flat", "dense"), \
            "{} not a valid graph type".format(graph_type)
        self.graph_type = graph_type
        super(Trace, self).__init__(*args, **kwargs)

    def add_node(self, site_name, *args, **kwargs):
        """
        :param string site_name: the name of the site to be added

        Adds a site to the trace.

        Identical to super(Trace, self).add_node,
        but raises an error when attempting to add a duplicate node
        instead of silently overwriting.
        """
        # XXX should do more validation than this
        if kwargs["type"] != "param":
            assert site_name not in self, \
                "site {} already in trace".format(site_name)

        # XXX should copy in case site gets mutated, or dont bother?
        super(Trace, self).add_node(site_name, *args, **kwargs.copy())

    def copy(self):
        """
        Makes a shallow copy of self with nodes and edges preserved.
        Identical to super(Trace, self).copy(), but preserves the type
        and the self.graph_type attribute
        """
        return Trace(self, graph_type=self.graph_type)

    def log_pdf(self, site_filter=lambda name, site: True):
        """
        Compute the local and overall log-probabilities of the trace.

        The local computation is memoized.
        """
        log_p = 0.0
        for name, site in self.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                try:
                    log_p += site["log_pdf"]
                except KeyError:
                    args, kwargs = site["args"], site["kwargs"]
                    site["log_pdf"] = site["fn"].log_pdf(
                        site["value"], *args, **kwargs) * site["scale"]
                    log_p += site["log_pdf"]
        return log_p

    def batch_log_pdf(self, site_filter=lambda name, site: True):
        """
        Compute the batched local and overall log-probabilities of the trace.

        The local computation is memoized, and also stores the local `.log_pdf()`.
        """
        log_p = 0.0
        # XXX will this iterate over nodes?
        for name, site in self.nodes.items():
            if site["type"] == "sample" and site_filter(name, site):
                try:
                    log_p += site["batch_log_pdf"]
                except KeyError:
                    args, kwargs = site["args"], site["kwargs"]
                    site["batch_log_pdf"] = site["fn"].batch_log_pdf(
                        site["value"], *args, **kwargs) * site["scale"]
                    site["log_pdf"] = site["batch_log_pdf"].sum()
                    log_p += site["batch_log_pdf"]
        return log_p

    @property
    def observation_nodes(self):
        """
        Gets a list of names of observe sites
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                node["obs"] is not None]

    @property
    def stochastic_nodes(self):
        """
        Gets a list of names of sample sites
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                node["obs"] is None]

    @property
    def reparameterized_nodes(self):
        """
        Gets a list of names of sample sites whose stochastic functions
        are reparameterizable primitive distributions
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                node["obs"] is None and
                getattr(node["fn"], "reparameterized", False)]

    @property
    def nonreparam_stochastic_nodes(self):
        """
        Gets a list of names of sample sites whose stochastic functions
        are not reparameterizable primitive distributions
        """
        return list(set(self.stochastic_nodes) - set(self.reparameterized_nodes))
