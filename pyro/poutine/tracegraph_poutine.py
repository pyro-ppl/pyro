import pyro
import torch
from torch.autograd import Variable, Function
import graphviz
import networkx
from collections import defaultdict
from .trace_poutine import TracePoutine


class TraceGraph(object):
    """
    -- encapsulates the forward graph as well as the trace of a stochastic function,
       along with some helper functions to access different node types.
    -- returned by TraceGraphPoutine
    -- visualization handled by save_visualization()
    """
    def __init__(self, G, trace, stochastic_nodes, reparameterized_nodes,
                 param_nodes, observation_nodes):
        self.G = G
        self.trace = trace
        self.param_nodes = param_nodes
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
        -- parameter nodes are light blue
        -- non-reparameterized stochastic nodes are salmon
        -- reparameterized stochastic nodes are half salmon, half grey
        -- observation nodes are green
        """
        g = graphviz.Digraph()
        for label in self.G.nodes():
            shape = 'ellipse'
            if label in self.param_nodes:
                fillcolor = 'lightblue'
            elif label in self.stochastic_nodes and label not in self.reparameterized_nodes:
                fillcolor = 'salmon'
            elif label in self.reparameterized_nodes:
                fillcolor = 'lightgrey;.5:salmon'
            elif label in self.observation_nodes:
                fillcolor = 'darkolivegreen3'
            g.node(label, label=label, shape=shape, style='filled', fillcolor=fillcolor)

        for label1, label2 in self.G.edges():
            g.edge(label1, label2)

        g.render(graph_output, view=False, cleanup=True)


class TraceGraphPoutine(TracePoutine):
    """
    trace graph poutine, used to generate a TraceGraph
    -- currently only supports 'coarse' graphs, i.e. overly-conservative ones constructed
       by following the sequential ordering of the execution trace
    TODO: add map data constructs
    """
    def __init__(self, fn, graph_type='coarse', include_params=False):
        assert(graph_type == 'coarse'), "only coarse graph type supported at present"
        super(TraceGraphPoutine, self).__init__(fn)
        self.include_params = include_params

    def _enter_poutine(self, *args, **kwargs):
        """
        enter and set up data structures
        """
        super(TraceGraphPoutine, self)._enter_poutine(*args, **kwargs)
        self.stochastic_nodes = []
        self.reparameterized_nodes = []
        self.param_nodes = []
        self.observation_nodes = []
        self.prev_node = '___ROOT_NODE___'
        self.G = networkx.DiGraph()

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Return a TraceGraph object that contains the forward graph and trace
        """
        self.trace = super(TraceGraphPoutine, self)._exit_poutine(ret_val, *args, **kwargs)
        self.G.remove_node('___ROOT_NODE___')

        trace_graph = TraceGraph(self.G, self.trace,
                                 self.stochastic_nodes, self.reparameterized_nodes,
                                 self.param_nodes, self.observation_nodes)
        return trace_graph

    def _add_graph_node(self, name):
        self.G.add_edge(self.prev_node, name)
        for ancestor in networkx.ancestors(self.G, self.prev_node):
            self.G.add_edge(ancestor, name)
        self.prev_node = name

    def _pyro_sample(self, msg, name, dist, *args, **kwargs):
        """
        register sampled variable for coarse graph construction
        """
        val = super(TraceGraphPoutine, self)._pyro_sample(msg, name, dist,
                                                          *args, **kwargs)
        self._add_graph_node(name)
        self.stochastic_nodes.append(name)
        if dist.reparameterized:
            self.reparameterized_nodes.append(name)
        return val

    def _pyro_param(self, msg, name, *args, **kwargs):
        """
        register parameter for coarse graph construction
        """
        retrieved = super(TraceGraphPoutine, self)._pyro_param(msg, name,
                                                               *args, **kwargs)
        if self.include_params:
            self._add_graph_node(name)
            self.param_nodes.append(name)
        return retrieved

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        register observe dependencies for coarse graph construction
        """
        val = super(TraceGraphPoutine, self)._pyro_observe(msg, name, fn, obs,
                                                           *args, **kwargs)
        self.observation_nodes.append(name)
        self._add_graph_node(name)
        return val
