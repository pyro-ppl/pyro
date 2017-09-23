import graphviz
import networkx
from collections import defaultdict, OrderedDict
from copy import copy

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
    """
    def __init__(self, fn, graph_type='coarse'):
        assert(graph_type == 'coarse'), "only coarse graph type supported at present"
        super(TraceGraphPoutine, self).__init__(fn)

    def _enter_poutine(self, *args, **kwargs):
        """
        enter and set up data structures
        """
        super(TraceGraphPoutine, self)._enter_poutine(*args, **kwargs)
        self.stochastic_nodes = []
        self.reparameterized_nodes = []
        self.observation_nodes = []
        self.prev_node = '___ROOT_NODE___'
        self.prev_nodes = {}
        self.prev_nodes_contexts = {}
        self.split_attach_points = {}
        self.split_node_counter = defaultdict(lambda: int())
        self.split_node_attached = defaultdict(lambda: False)
        self.G = networkx.DiGraph()

    def _attach_join_node(self, join_node, split_node):
        parents = self.G.predecessors(split_node)
        while True:
            assert(len(parents) < 2)
            if len(parents) == 0:
                return
            elif parents[0][-12:] != '__SPLIT_NODE':
                children = self.G.successors(parents[0])
                target_child = list(filter(lambda n: n[-5:] != '_NODE', children))
                assert(len(target_child) <= 1)
                if len(target_child) == 1:
                    self.G.add_edge(join_node, target_child[0])
                return
            parents = self.G.predecessors(parents[0])

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Return a TraceGraph object that contains the forward graph and trace
        """
        self.trace = super(TraceGraphPoutine, self)._exit_poutine(ret_val, *args, **kwargs)

        # in the presence of map data there may be dangling join nodes that need to be
        # stitched appropriately. do that here
        print "SPLIT ATTACH POINTS"
        for k in self.split_attach_points:
            break
            print "[%s]" % k, self.split_attach_points[k]
            if len(self.G.predecessors(k))==0:
                self.G.add_edge(self.split_attach_points[k], k)
        print "***************"

        for split_node in filter(lambda n: n[-12:] == '__SPLIT_NODE', self.G.nodes()):
            break
            join_node = split_node[:-10] + 'JOIN_NODE'
            self._attach_join_node(join_node, split_node)

        # remove all auxiliary nodes while preserving dependency structure
        for node in self.G.nodes():
            break
            if node[-12:] == '__SPLIT_NODE':
                parents = self.G.predecessors(node)
                assert(len(parents) == 1)
                children = self.G.successors(node)
                for c in children:
                    for p in parents:
                        self.G.add_edge(p, c)
                self.G.remove_node(node)
        for node in self.G.nodes():
            break
            if node[-11:] == '__JOIN_NODE':
                parents = self.G.predecessors(node)
                children = self.G.successors(node)
                for c in children:
                    for p in parents:
                        self.G.add_edge(p, c)
                self.G.remove_node(node)
        #if '___ROOT_NODE___' in self.G.nodes():
        #    self.G.remove_node('___ROOT_NODE___')

        # make sure all dependencies are explicitly accounted for
        # (up to this point the graph structure is a skeleton in which some
        # dependencies haven't been made explicit)
        # XXX can we do this better given the graph structure at this point?
        #for node in self.G.nodes():
        #    for ancestor in networkx.ancestors(self.G, node):
        #        self.G.add_edge(ancestor, node)

        trace_graph = TraceGraph(self.G, self.trace,
                                 self.stochastic_nodes, self.reparameterized_nodes,
                                 self.observation_nodes)
        return trace_graph

    def _get_prev_node(self, stack, context):
        if len(stack)==0:
            return self.prev_node
        if stack not in self.prev_nodes_contexts or \
            (stack in self.prev_nodes_contexts and self.prev_nodes_contexts[stack] != context):
            self.prev_nodes[stack] = stack[0] + "__SPLIT_NODE"
            self.prev_nodes_contexts[stack] = context
        return self.prev_nodes[stack]

    def _set_prev_node(self, stack, new_value, context):
        if len(stack)==0:
            self.prev_node = new_value
        else:
            self.prev_nodes[stack] = new_value
        self.prev_nodes_contexts[stack] = context

    def _attach_split_node(self, split_node, join_node, stack, context):
        prev_node = self._get_prev_node(stack, context)
        self.G.add_edge(prev_node, split_node)
        self.split_node_attached[split_node] = True
        self._set_prev_node(stack, join_node, context)
        if prev_node[-12:] == '__SPLIT_NODE' and prev_node not in self.split_node_attached:
            prev_join_node = prev_node[:-12] + '__JOIN_NODE'
            self.G.add_edge(join_node, prev_join_node)
            self._attach_split_node(prev_node, prev_join_node, stack[1:], stack[0])

    def _add_graph_node(self, msg, name):
        map_data_stack = tuple(msg['map_data_stack'])
        map_data_stack_height = len(map_data_stack)

        if map_data_stack_height > 0:
            nodes = msg['map_data_nodes']
            if nodes['previous'] == nodes['split']:
                self.G.add_edge(nodes['split'], nodes['current'])
                #self.G.add_edge(nodes['split'], nodes['join'])
                if not self.split_node_attached[nodes['split']]:
                    self._attach_split_node(nodes['split'], nodes['join'],
                                            map_data_stack[1:], map_data_stack[0])
            else:
                self.G.add_edge(nodes['previous'], nodes['current'])
            #self._set_prev_node(map_data_stack[1:], self.split_node_counter[nodes['split']],
            #        nodes['join'])
            self.G.add_edge(nodes['current'], nodes['join'])
            #self._set_prev_node(stack, nodes['current'], ())
            #self._set_prev_node(map_data_stack[1:], nodes['join'])
            #if nodes['previous'] == nodes['split']:
            #    print "%s == %s" % (nodes['previous'], nodes['split'])
                #if nodes['split'] not in self.split_attach_points:
                #self.split_attach_points[nodes['split']] = self._get_prev_node(map_data_stack[1:])
                #self.G.add_edge(nodes['split'], self._get_prev_node(map_data_stack[1:]))
                #self._set_prev_node(map_data_stack[1:], nodes['join'])
            #for k in range(1, map_data_stack_height):
            #    split_node_k_1 = map_data_stack[k - 1] + '__SPLIT_NODE'
            #    split_node_k = map_data_stack[k] + '__SPLIT_NODE'
            #    self.G.add_edge(split_node_k, split_node_k_1)
                #for k, map_data_node in enumerate(map_data_stack[1:]):
                #    if map_data_node + '__SPLIT_NODE' not in self.split_attach_points and k<len(map_data_stack[1:])-1:
                #        self.split_attach_points[map_data_node] = map_data_stack[k+2] + '__SPLIT_NODE'
        else:
            self.G.add_edge(self.prev_node, name)
            self.prev_node = name

    def _pyro_sample(self, msg, name, dist, *args, **kwargs):
        """
        register sampled variable for coarse graph construction
        """
        val = super(TraceGraphPoutine, self)._pyro_sample(msg, name, dist,
                                                          *args, **kwargs)
        self._add_graph_node(msg, name)
        self.stochastic_nodes.append(name)
        if dist.reparameterized:
            self.reparameterized_nodes.append(name)
        return val

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        register observe dependencies for coarse graph construction
        """
        val = super(TraceGraphPoutine, self)._pyro_observe(msg, name, fn, obs,
                                                           *args, **kwargs)
        self._add_graph_node(msg, name)
        self.observation_nodes.append(name)
        return val

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None, batch_dim=0):
        ret = super(TraceGraphPoutine, self)._pyro_map_data(msg, name, data, fn,
                                                            batch_size=batch_size,
                                                            batch_dim=batch_dim)
        return ret
