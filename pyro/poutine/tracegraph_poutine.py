import networkx

from collections import defaultdict
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
        Construct a TraceGraph object that contains the forward graph and trace
        """
        vectorized_map_data_info = self._get_vectorized_map_data_info()
        self.trace_graph = TraceGraph(self.G, self.trace,
                                      self.stochastic_nodes,
                                      self.reparameterized_nodes,
                                      self.observation_nodes, vectorized_map_data_info)
        return super(TraceGraphPoutine, self).__exit__(*args)

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace_graph

    def _get_vectorized_map_data_info(self):
        """
        this determines whether the vectorized map_datas are rao-blackwellizable by tracegraph_kl_qp
        also gathers information to be consumed by downstream by tracegraph_kl_qp
        XXX this logic should probably sit elsewhere
        """
        vectorized_map_data_info = {'rao-blackwellization-condition': True}
        vec_md_stacks = set()

        for node, stack in self.nodes_seen_so_far.items():
            vec_mds = list(filter(lambda x: x[2] == 'tensor', stack))
            # check for nested vectorized map datas
            if len(vec_mds) > 1:
                vectorized_map_data_info['rao-blackwellization-condition'] = False
            # check that vectorized map datas only found at innermost position
            if len(vec_mds) > 0 and stack[-1][2] == 'list':
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
                vec_mds = list(filter(lambda x: x[2] == 'tensor', stack))
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

    def _pyro_sample(self, msg):
        """
        register sample dependencies for coarse graph construction
        """
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
