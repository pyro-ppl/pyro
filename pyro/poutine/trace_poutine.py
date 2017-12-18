from __future__ import absolute_import, division, print_function

from .poutine import Poutine
from .trace import Trace
from .util import site_is_subsample


def get_vectorized_map_data_info(trace):
    """
    This determines whether the vectorized map_datas are rao-blackwellizable by
    `TraceGraph_ELBO`. This also gathers information to be consumed by
    downstream by `TraceGraph_ELBO`.
    """
    nodes = trace.nodes

    vectorized_map_data_info = {'rao-blackwellization-condition': True, 'warnings': set()}
    vec_md_stacks = set()

    for name, node in nodes.items():
        if site_is_subsample(node):
            continue
        if node["type"] in ("sample", "param"):
            stack = tuple(node["cond_indep_stack"])
            vec_mds = [x for x in stack if x.vectorized]
            # check for nested vectorized map datas
            if len(vec_mds) > 1:
                vectorized_map_data_info['rao-blackwellization-condition'] = False
                vectorized_map_data_info['warnings'].add('nested iarange')
            # check that vectorized map datas only found at innermost position
            if vec_mds and not stack[-1].vectorized:
                vectorized_map_data_info['rao-blackwellization-condition'] = False
                vectorized_map_data_info['warnings'].add('non-leaf iarange')
            # enforce that if there are multiple vectorized map_datas, they are all
            # independent of one another because of enclosing list map_datas
            # (step 1: collect the stacks)
            if vec_mds:
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
                    if md_i.name == md_j.name and md_i.counter != md_j.counter:
                        ij_independent = True
                if not ij_independent:
                    vectorized_map_data_info['rao-blackwellization-condition'] = False
                    vectorized_map_data_info['warnings'].add('there exist dependent iaranges')
                    break

    # construct data structure consumed by tracegraph_kl_qp
    if vectorized_map_data_info['rao-blackwellization-condition']:
        vectorized_map_data_info['nodes'] = set()
        for name, node in nodes.items():
            if site_is_subsample(node):
                continue
            if node["type"] in ("sample", "param"):
                if any(x.vectorized for x in node["cond_indep_stack"]):
                    vectorized_map_data_info['nodes'].add(name)

    return vectorized_map_data_info


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


class TracePoutine(Poutine):
    """
    Execution trace poutine.

    A TracePoutine records the input and output to every pyro primitive
    and stores them as a site in a Trace().
    This should, in theory, be sufficient information for every inference algorithm
    (along with the implicit computational graph in the Variables?)

    We can also use this for visualization.
    """

    def __init__(self, fn, graph_type=None):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)
        :param string graph_type: string that specifies the type of graph
            to construct (currently only "flat" or "dense" supported)
        """
        if graph_type is None:
            graph_type = "flat"
        assert graph_type in ("flat", "dense")
        self.graph_type = graph_type
        super(TracePoutine, self).__init__(fn)

    def __exit__(self, *args, **kwargs):
        """
        Adds appropriate edges based on cond_indep_stack information
        upon exiting the context.
        """
        if self.graph_type == "dense":
            identify_dense_edges(self.trace)
            self.trace.graph["vectorized_map_data_info"] = \
                get_vectorized_map_data_info(self.trace)
        return super(TracePoutine, self).__exit__(*args, **kwargs)

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
        self.trace = Trace(graph_type=self.graph_type)
        self.trace.add_node("_INPUT",
                            name="_INPUT", type="args",
                            args=args, kwargs=kwargs)
        ret = super(TracePoutine, self).__call__(*args, **kwargs)
        self.trace.add_node("_RETURN", name="_RETURN", type="return", value=ret)
        return ret

    def get_trace(self, *args, **kwargs):
        """
        :returns: data structure
        :rtype: pyro.poutine.Trace

        Helper method for a very common use case.
        Calls this poutine and returns its trace instead of the function's return value.
        """
        self(*args, **kwargs)
        return self.trace.copy()

    def _reset(self):
        tr = Trace(graph_type=self.graph_type)
        tr.add_node("_INPUT",
                    name="_INPUT", type="input",
                    args=self.trace.nodes["_INPUT"]["args"],
                    kwargs=self.trace.nodes["_INPUT"]["kwargs"])
        self.trace = tr
        super(TracePoutine, self)._reset()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.

        Implements default pyro.sample Poutine behavior with an additional side effect:
        if the observation at the site is not None,
        then store the observation in self.trace
        and return the observation,
        else call the function,
        then store the return value in self.trace
        and return the return value.
        """
        name = msg["name"]
        if name in self.trace:
            site = self.trace.nodes[name]
            if site['type'] == 'param':
                # Cannot sample or observe after a param statement.
                raise RuntimeError("{} is already in the trace as a param".format(name))
            elif site['type'] == 'sample':
                # Cannot sample after a previous sample statement.
                raise RuntimeError("Multiple pyro.sample sites named '{}'".format(name))

        val = super(TracePoutine, self)._pyro_sample(msg)
        site = msg.copy()
        site.update(value=val)
        self.trace.add_node(name, **site)
        return val

    def _pyro_param(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the result of querying the parameter store

        Implements default pyro.param Poutine behavior with an additional side effect:
        queries the parameter store with the site name and varargs
        and returns the result of the query.

        If the parameter doesn't exist, create it using the site varargs.
        If it does exist, grab it from the parameter store.
        Store the parameter in self.trace, and then return the parameter.
        """
        if msg["name"] in self.trace:
            if self.trace.nodes[msg['name']]['type'] == "sample":
                raise RuntimeError("{} is already in the trace as a sample".format(msg['name']))

        val = super(TracePoutine, self)._pyro_param(msg)
        site = msg.copy()
        site.update(value=val)
        self.trace.add_node(msg["name"], **site)
        return val
