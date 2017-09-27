import graphviz
import networkx

from .trace_poutine import TracePoutine
from collections import defaultdict

class TraceGraph(object):
    """
    -- encapsulates the forward graph as well as the trace of a stochastic function,
       along with some helper functions to access different node types.
    -- returned by TraceGraphPoutine
    -- visualization handled by save_visualization()
    """

    def __init__(self, G, trace, stochastic_nodes, reparameterized_nodes,
                 observation_nodes,
                 vectorized_map_data_info):
        self.G = G
        self.trace = trace
        self.reparameterized_nodes = reparameterized_nodes
        self.stochastic_nodes = stochastic_nodes
        self.nonreparam_stochastic_nodes = list(set(stochastic_nodes) - set(reparameterized_nodes))
        self.observation_nodes = observation_nodes
        #self.map_data_stacks = map_data_stacks
        self.vectorized_map_data_info = vectorized_map_data_info

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

        guide_tracegraph = poutine.tracegraph(guide)(*args, **kwargs)
        guide_tracegraph.save_visualization('guide')
        guide_trace = guide_tracegraph.get_trace()
        model_tracegraph = poutine.tracegraph(poutine.replay(model, guide_trace))(*args, **kwargs)
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

    def _enter_poutine(self, *args, **kwargs):
        """
        enter and set up data structures
        """
        super(TraceGraphPoutine, self)._enter_poutine(*args, **kwargs)
        self.stochastic_nodes = []
        self.reparameterized_nodes = []
        self.observation_nodes = []
        self.nodes_seen_so_far = {}
        self.G = networkx.DiGraph()

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Return a TraceGraph object that contains the forward graph and trace
        """
        self.trace = super(TraceGraphPoutine, self)._exit_poutine(ret_val, *args, **kwargs)

        vectorized_map_data_info = self._get_vectorized_map_data_info()

        trace_graph = TraceGraph(self.G, self.trace,
                                 self.stochastic_nodes, self.reparameterized_nodes,
                                 self.observation_nodes,
                                 vectorized_map_data_info)
        return trace_graph

    def _get_vectorized_map_data_info(self):
        """
        this determines whether the vectorized map_datas are rao-blackwellizable by tracegraph_kl_qp
        also gathers information to be consumed by downstream by tracegraph_kl_qp
        XXX this logic should probably sit elsewhere
        """
        vectorized_map_data_info = {'rao-blackwellization-condition': True}
        vec_md_stacks = set()

        for node, stack in self.nodes_seen_so_far.items():
            vec_mds = filter(lambda x: x[2]=='tensor', stack)
            # check for nested vectorized map datas
            if len(vec_mds) > 1:
                vectorized_map_data_info['rao-blackwellization-condition'] = False
            # check that vectorized map datas only found at innermost position
            if len(vec_mds) > 0 and stack[-1][2]=='list':
                vectorized_map_data_info['rao-blackwellization-condition'] = False
            # for now enforce batch_dim = 0 for vectorized map_data
            # since needed batch_log_pdf infrastructure missing
            if len(vec_mds) > 0 and vec_mds[0][3] != 0:
                vectorized_map_data_info['rao-blackwellization-condition'] = False
            # enforce that if there are multiple vectorized map_datas, they are all
            # independent of one another because of enclosing list map_datas
            # (step 1: collect the stacks)
            if len(vec_mds) > 0:
                vec_md_stacks.add(stack)
            # bail, since condition false
            if not vectorized_map_data_info['rao-blackwellization-condition']:
                break

        # enforce that if there are multiple vectorized map_datas, they are all
        # independent of one another because of enclosing list map_datas
        # (step 2: explicitly check this)
        if vectorized_map_data_info['rao-blackwellization-condition']:
            vec_md_stacks = list(vec_md_stacks)
            for i, stack_i in enumerate(vec_md_stacks):
                for j, stack_j in enumerate(vec_md_stacks):
                    # only check unique pairs
                    if i <= j:
                        continue
                    ij_independent = False
                    for md_i, md_j in zip(stack_i, stack_j):
                        if md_i[0] == md_j[0] and md_i[1] != md_j[1]:
                            ij_independent = True
                    if not ij_independent:
                        vectorized_map_data_info['rao-blackwellization-condition'] = False
                        break

        # construct data structure consumed by tracegraph_kl_qp
        if vectorized_map_data_info['rao-blackwellization-condition']:
            vectorized_map_data_info['nodes'] = defaultdict(lambda: [])
            for node, stack in self.nodes_seen_so_far.items():
                vec_mds = filter(lambda x: x[2]=='tensor', stack)
                if len(vec_mds) > 0:
                    node_batch_dim_pair = (node, vec_mds[0][3])
                    vectorized_map_data_info['nodes'][vec_mds[0][0]].append(node_batch_dim_pair)

        return vectorized_map_data_info

    def _add_graph_node(self, msg, name):
        """
        used internally to attach the node at the current site to any nodes
        that it could depend on. 90% of the logic is for making sure independencies
        from (list) map_data are taken into account. note that this has potentially
        bad asymptotics because the result is a (possibly) very dense graph
        """
        map_data_stack = tuple(reversed(msg['map_data_stack']))

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

    def _pyro_sample(self, msg, name, dist, *args, **kwargs):
        """
        register sample dependencies for coarse graph construction
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
