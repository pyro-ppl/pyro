import graphviz
import networkx

from .trace_poutine import TracePoutine


class TraceGraph(object):
    """
    -- encapsulates the forward graph as well as the trace of a stochastic function,
       along with some helper functions to access different node types.
    -- returned by TraceGraphPoutine
    -- visualization handled by save_visualization()
    """

    def __init__(self, G, trace, stochastic_nodes, reparameterized_nodes,
                 observation_nodes):
        self.G = G
        self.trace = trace
        self.reparameterized_nodes = reparameterized_nodes
        self.stochastic_nodes = stochastic_nodes
        self.nonreparam_stochastic_nodes = list(set(stochastic_nodes) - set(reparameterized_nodes))
        self.observation_nodes = observation_nodes

    def get_stochastic_nodes(self):
        """
        get all sample nodes in graph
        """
        return self.stochastic_nodes

    def get_nonreparam_stochastic_nodes(self):
        """
        get all non-reparameterized sample nodes in graph
        """
        return self.nonreparam_stochastic_nodes

    def get_reparam_stochastic_nodes(self):
        """
        get all reparameterized sample nodes in graph
        """
        return self.reparameterized_nodes

    def get_nodes(self):
        """
        get all nodes in graph
        """
        return self.G.nodes()

    def get_children(self, node, with_self=False):
        """
        get children of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the children
        """
        children = self.G.successors(node)
        if with_self:
            children.append(node)
        return children

    def get_parents(self, node, with_self=False):
        """
        get parents of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the parents
        """
        parents = self.G.predecessors(node)
        if with_self:
            parents.append(node)
        return parents

    def get_ancestors(self, node, with_self=False):
        """
        get ancestors of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the ancestors
        """
        ancestors = list(networkx.ancestors(self.G, node))
        if with_self:
            ancestors.append(node)
        return ancestors

    def get_descendants(self, node, with_self=False):
        """
        get descendants of a named node
        :param node: the name of the node in the tracegraph
        :param with_self: whether to include `node` among the descendants
        """
        descendants = list(networkx.descendants(self.G, node))
        if with_self:
            descendants.append(node)
        return descendants

    def get_trace(self):
        """
        get the Trace associated with the TraceGraph
        """
        return self.trace

    def get_graph(self):
        """
        get the graph associated with the TraceGraph
        """
        return self.G

    def save_visualization(self, graph_output):
        """
        render graph and save to file
        :param graph_output: the graph will be saved to graph_output.pdf
        -- non-reparameterized stochastic nodes are salmon
        -- reparameterized stochastic nodes are half salmon, half grey
        -- observation nodes are green
        """
        g = graphviz.Digraph()
        for label in self.G.nodes():
            shape = 'ellipse'
            if label in self.stochastic_nodes and label not in self.reparameterized_nodes:
                fillcolor = 'salmon'
            elif label in self.reparameterized_nodes:
                fillcolor = 'lightgrey;.5:salmon'
            elif label in self.observation_nodes:
                fillcolor = 'darkolivegreen3'
            else:
                fillcolor = 'grey'
            g.node(label, label=label, shape=shape, style='filled', fillcolor=fillcolor)

        for label1, label2 in self.G.edges():
            g.edge(label1, label2)

        g.render(graph_output, view=False, cleanup=True)


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
        name, fn, args, kwargs = \
            msg["name"], msg["fn"], msg["args"], msg["kwargs"]
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
        name, fn, obs, args, kwargs = \
            msg["name"], msg["fn"], msg["obs"], msg["args"], msg["kwargs"]
        val = super(TraceGraphPoutine, self)._pyro_observe(msg)
        self._add_graph_node(msg, name)
        self.observation_nodes.append(name)
        return val
