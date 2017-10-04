import networkx

from .trace_poutine import TracePoutine
from .trace import TraceGraph


class TraceGraphPoutine(TracePoutine):
    """
    trace graph poutine, used to generate a TraceGraph
    -- currently only supports 'coarse' graphs, i.e. overly-conservative ones constructed
       by following the sequential ordering of the execution trace
    TODO: add constructs for vectorized map data

    this can be invoked as follows to create visualizations of the TraceGraph:

        guide_tracegraph = poutine.tracegraph(guide).get_trace(*args, **kwargs)
        guide_tracegraph.save_visualization('guide')
        guide_trace = guide_tracegraph.get_trace()
        model_tracegraph = poutine.tracegraph(poutine.replay(model, guide_trace)).get_trace(*args, **kwargs)
        model_tracegraph.save_visualization('model')
        model_trace = model_tracegraph.get_trace()

    if the visualization proves difficult to parse, one can also directly interrogate the networkx
    graph object, e.g.:

        print model_tracegraph.get_graph().nodes()
        print model_tracegraph.get_graph().edges()

    """
    def __init__(self, fn, graph_type='coarse'):
        assert(graph_type == 'coarse'), "only coarse graph type supported at present"
        super(TraceGraphPoutine, self).__init__(fn)

    def __enter__(self):
        """
        enter and set up data structures
        """
        self.stochastic_nodes = []
        self.reparameterized_nodes = []
        self.observation_nodes = []
        self.nodes_seen_so_far = {}
        self.G = networkx.DiGraph()
        return super(TraceGraphPoutine, self).__enter__()

    def __exit__(self, *args):
        """
        OUTDATED Return a TraceGraph object that contains the forward graph and trace
        """
        self.trace_graph = TraceGraph(self.G, self.trace,
                                      self.stochastic_nodes,
                                      self.reparameterized_nodes,
                                      self.observation_nodes)
        return super(TraceGraphPoutine, self).__exit__(*args)

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace_graph

    def _add_graph_node(self, msg, name):
        """
        used internally to attach the node at the current site to any nodes
        that it could depend on. 90% of the logic is for making sure independencies
        from (list) map_data are taken into account. note that this has bad asymptotics
        because the result is a (possibly) very dense graph
        """
        map_data_stack = list(reversed(msg['map_data_stack']))

        # for each node seen thus far we determine whether the current
        # node should depend on it. in this context the answer is always yes
        # unless map_data is telling us otherwise
        for node in self.nodes_seen_so_far:
            node_independent = False
            node_map_data_stack = self.nodes_seen_so_far[node]
            for query, target in zip(map_data_stack, node_map_data_stack):
                if query[0] == target[0] and query[1] != target[1]:
                    node_independent = True
                    break
            if not node_independent:
                self.G.add_edge(node, name)

        self.G.add_node(name)
        self.nodes_seen_so_far[name] = map_data_stack

    def _pyro_sample(self, msg):
        """
        register sample dependencies for coarse graph construction
        """
        # TODO remove unnecessary
        name, fn = \
            msg["name"], msg["fn"]
        val = super(TraceGraphPoutine, self)._pyro_sample(msg)
        self._add_graph_node(msg, name)
        self.stochastic_nodes.append(name)
        if hasattr(fn, "reparameterized") and fn.reparameterized:
            self.reparameterized_nodes.append(name)
        return val

    def _pyro_observe(self, msg):
        """
        register observe dependencies for coarse graph construction
        """
        name = msg["name"]
        val = super(TraceGraphPoutine, self)._pyro_observe(msg)
        self._add_graph_node(msg, name)
        self.observation_nodes.append(name)
        return val
