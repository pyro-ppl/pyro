import collections
import networkx

from pyro import util


class Trace(networkx.DiGraph):
    """
    Execution trace data structure
    """

    node_dict_factory = collections.OrderedDict

    def __init__(self, *args, **kwargs):
        """
        TODO docs
        constructor
        """
        graph_type = kwargs.pop("graph_type", "flat")
        assert graph_type in ("flat", "dense"), \
            "{} not a valid graph type".format(graph_type)
        self.graph_type = graph_type
        super(Trace, self).__init__(*args, **kwargs)

    def add_node(self, site_name, *args, **kwargs):
        """
        TODO docs
        add site
        """
        # XXX should do more validation than this
        if kwargs["type"] != "param":
            assert site_name not in self, \
                "site {} already in trace".format(site_name)

        # XXX should copy in case site gets mutated, or dont bother?
        super(Trace, self).add_node(site_name, *args, **kwargs.copy())

    def identify_edges(self):
        """
        TODO docs
        """
        if self.graph_type == "dense":
            # XXX will this iterate over nodes?
            for name, node in self.nodes.items():
                if node["type"] in ("sample", "observe", "param"):
                    # XXX why tuple?
                    map_data_stack = tuple(reversed(node["map_data_stack"]))
                    for past_name, past_node in self.nodes.items():
                        if past_node["type"] in ("sample", "observe", "param"):
                            if past_name == name:
                                break
                            past_node_independent = False
                            past_node_map_data_stack = tuple(
                                reversed(past_node["map_data_stack"]))
                            for query, target in zip(map_data_stack,
                                                     past_node_map_data_stack):
                                if query[0] == target[0] and query[1] != target[1]:
                                    past_node_independent = True
                                    break
                            if not past_node_independent:
                                self.add_edge(past_name, name)

            self.vectorized_map_data_info = util.get_vectorized_map_data_info(self.nodes)

    def copy(self):
        cp = Trace(self, graph_type=self.graph_type)
        cp.vectorized_map_data_info = util.get_vectorized_map_data_info(cp.nodes)
        return cp

    def log_pdf(self, site_filter=lambda name, site: True):
        """
        Compute the local and overall log-probabilities of the trace.

        The local computation is memoized.
        """
        log_p = 0.0
        for name, site in self.nodes.items():
            if site["type"] in ("observe", "sample") and site_filter(name, site):
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
            if site["type"] in ("observe", "sample") and site_filter(name, site):
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
        TODO docs
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "observe"]

    @property
    def stochastic_nodes(self):
        """
        TODO docs
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample"]

    @property
    def reparameterized_nodes(self):
        """
        TODO docs
        """
        return [name for name, node in self.nodes.items()
                if node["type"] == "sample" and
                getattr(node["fn"], "reparameterized", False)]

    @property
    def nonreparam_stochastic_nodes(self):
        """
        TODO docs
        """
        return list(set(self.stochastic_nodes) - set(self.reparameterized_nodes))
