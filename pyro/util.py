import pyro
import graphviz
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter
from collections import defaultdict


def detach_iterable(iterable):
    if isinstance(iterable, Variable):
        return iterable.detach()
    else:
        return [var.detach() for var in iterable]


def _dict_to_tuple(d):
    """
    Recursively converts a dictionary to a list of key-value tuples
    Only intended for use as a helper function inside memoize!!
    May break when keys cant be sorted, but that is not an expected use-case
    """
    if isinstance(d, dict):
        return tuple([(k, _dict_to_tuple(d[k])) for k in sorted(d.keys())])
    else:
        return d


def memoize(fn):
    """
    https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
    unbounded memoize
    alternate in py3: https://docs.python.org/3/library/functools.html
    lru_cache
    """
    mem = {}

    def _fn(*args, **kwargs):
        kwargs_tuple = _dict_to_tuple(kwargs)
        if (args, kwargs_tuple) not in mem:
            mem[(args, kwargs_tuple)] = fn(*args, **kwargs)
        return mem[(args, kwargs_tuple)]
    return _fn


def set_rng_seed(rng_seed):
    """
    Sets seeds of torch, numpy, and torch.cuda (if available).
    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    np.random.seed(rng_seed)


def ones(*args, **kwargs):
    return Parameter(torch.ones(*args, **kwargs))


def zeros(*args, **kwargs):
    return Parameter(torch.zeros(*args, **kwargs))


def ng_ones(*args, **kwargs):
    return Variable(torch.ones(*args, **kwargs), requires_grad=False)


def ng_zeros(*args, **kwargs):
    return Variable(torch.zeros(*args, **kwargs), requires_grad=False)


def log_sum_exp(vecs):
    n = len(vecs.size())
    if n == 1:
        vecs = vecs.view(1, -1)
    _, idx = torch.max(vecs, 1)
    max_score = torch.index_select(vecs, 1, idx.view(-1))
    ret = max_score + torch.log(torch.sum(torch.exp(vecs - max_score.expand_as(vecs))))
    if n == 1:
        return ret.view(-1)
    return ret


def zero_grads(tensors):
    """
    Sets gradients of list of Variables to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            if p.grad.volatile:
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


def log_gamma(xx):
    if isinstance(xx, Variable):
        ttype = xx.data.type()
    elif isinstance(xx, torch.Tensor):
        ttype = xx.type()
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = Variable(torch.ones(x.size()).type(ttype)) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def log_beta(t):
    """
    Computes log Beta function.

    :param t:
    :type t: torch.autograd.Variable of dimension 1 or 2
    :rtype: torch.autograd.Variable of float (if t.dim() == 1) or torch.Tensor (if t.dim() == 2)
    """
    assert t.dim() in (1, 2)
    if t.dim() == 1:
        numer = torch.sum(log_gamma(t))
        denom = log_gamma(torch.sum(t))
    else:
        numer = torch.sum(log_gamma(t), 1)
        denom = log_gamma(torch.sum(t, 1))
    return numer - denom


def to_one_hot(x, ps):
    if isinstance(x, Variable):
        ttype = x.data.type()
    elif isinstance(x, torch.Tensor):
        ttype = x.type()
    batch_size = x.size(0)
    classes = ps.size(1)
    # create an empty array for one-hots
    batch_one_hot = torch.zeros(batch_size, classes)
    # this operation writes ones where needed
    batch_one_hot.scatter_(1, x.data.view(-1, 1).long(), 1)

    return Variable(batch_one_hot.type(ttype))


def tensor_histogram(ps, vs):
    """
    make a histogram from weighted Variable/Tensor/ndarray samples
    Horribly slow...
    """
    # first, get everything into the same form: numpy arrays
    np_vs = []
    for v in vs:
        _v = v
        if isinstance(_v, Variable):
            _v = _v.data
        if isinstance(_v, torch.Tensor):
            _v = _v.numpy()
        np_vs.append(_v)
    # now form the histogram
    hist = dict()
    for p, v, np_v in zip(ps, vs, np_vs):
        k = tuple(np_v.flatten().tolist())
        if k not in hist:
            # XXX should clone?
            hist[k] = [0.0, v]
        hist[k][0] = hist[k][0] + p
    # now split into keys and original values
    ps2 = []
    vs2 = []
    for k in hist.keys():
        ps2.append(hist[k][0])
        vs2.append(hist[k][1])
    # return dict suitable for passing into Categorical
    return {"ps": torch.cat(ps2), "vs": np.array(vs2).flatten()}


def basic_histogram(ps, vs):
    """
    make a histogram from weighted things that aren't tensors
    Horribly slow...
    """
    assert isinstance(vs, (list, tuple)), \
        "vs must be a primitive type that preserves ordering at construction"
    hist = {}
    for i, v in enumerate(vs):
        if v not in hist:
            hist[v] = 0.0
        hist[v] = hist[v] + ps[i]
    return {"ps": torch.cat([hist[v] for v in hist.keys()]),
            "vs": [v for v in hist.keys()]}


def apply_stack(initial_msg):
    """
    :param dict initial_msg: the starting version of the trace site
    :returns: an updated message that is the final version of the trace site

    Execute the poutine stack at a single site according to the following scheme:
    1. Walk down the stack from top to bottom, collecting into the message
        all information necessary to execute the stack at that site
    2. For each poutine in the stack from bottom to top:
           Execute the poutine with the message;
           If the message field "stop" is True, stop;
           Otherwise, continue
    3. Return the updated message
    """
    stack = pyro._PYRO_STACK
    # TODO check at runtime if stack is valid

    # msg is used to pass information up and down the stack
    msg = initial_msg

    # first, gather all information necessary to apply the stack to this site
    for frame in reversed(stack):
        msg = frame._prepare_site(msg)

    # go until time to stop?
    for frame in stack:
        assert msg["type"] in ("sample", "observe", "map_data", "param"), \
            "{} is an invalid site type, how did that get there?".format(msg["type"])

        msg["value"] = getattr(frame, "_pyro_{}".format(msg["type"]))(msg)

        if msg["stop"]:
            break

    return msg


class NonlocalExit(Exception):
    """
    Exception for exiting nonlocally from poutine execution.

    Used by poutine.EscapePoutine to return site information.
    """
    def __init__(self, site, *args, **kwargs):
        """
        :param site: message at a pyro site

        constructor.  Just stores the input site.
        """
        super(NonlocalExit, self).__init__(*args, **kwargs)
        self.site = site


def enum_extend(trace, msg, num_samples=None):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :param num_samples: maximum number of extended traces to return.
    :returns: a list of traces, copies of input trace with one extra site

    Utility function to copy and extend a trace with sites based on the input site
    whose values are enumerated from the support of the input site's distribution.

    Used for exact inference and integrating out discrete variables.
    """
    if num_samples is None:
        num_samples = -1

    extended_traces = []
    for i, s in enumerate(msg["fn"].support(*msg["args"], **msg["kwargs"])):
        if i > num_samples and num_samples >= 0:
            break
        msg_copy = msg.copy()
        msg_copy.update(value=s)
        tr_cp = trace.copy()
        tr_cp.add_node(msg["name"], **msg_copy)
        extended_traces.append(tr_cp)
    return extended_traces


def mc_extend(trace, msg, num_samples=None):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :param num_samples: maximum number of extended traces to return.
    :returns: a list of traces, copies of input trace with one extra site

    Utility function to copy and extend a trace with sites based on the input site
    whose values are sampled from the input site's function.

    Used for Monte Carlo marginalization of individual sample sites.
    """
    if num_samples is None:
        num_samples = 1

    extended_traces = []
    for i in range(num_samples):
        msg_copy = msg.copy()
        msg_copy["value"] = msg_copy["fn"](*msg_copy["args"], **msg_copy["kwargs"])
        tr_cp = trace.copy()
        tr_cp.add_node(msg_copy["name"], **msg_copy)
        extended_traces.append(tr_cp)
    return extended_traces


def discrete_escape(trace, msg):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :returns: boolean decision value

    Utility function that checks if a sample site is discrete and not already in a trace.

    Used by EscapePoutine to decide whether to do a nonlocal exit at a site.
    Subroutine for integrating out discrete variables for variance reduction.
    """
    return (msg["type"] == "sample") and \
        (msg["name"] not in trace) and \
        (getattr(msg["fn"], "enumerable", False))


def all_escape(trace, msg):
    """
    :param trace: a partial trace
    :param msg: the message at a pyro primitive site
    :returns: boolean decision value

    Utility function that checks if a site is not already in a trace.

    Used by EscapePoutine to decide whether to do a nonlocal exit at a site.
    Subroutine for approximately integrating out variables for variance reduction.
    """
    return (msg["type"] == "sample") and \
        (msg["name"] not in trace)


def get_vectorized_map_data_info(nodes):
    """
    this determines whether the vectorized map_datas
    are rao-blackwellizable by tracegraph_kl_qp
    also gathers information to be consumed by downstream by tracegraph_kl_qp
    XXX this logic should probably sit elsewhere
    """
    vectorized_map_data_info = {'rao-blackwellization-condition': True}
    vec_md_stacks = set()

    for name, node in nodes.items():
        if node["type"] in ("sample", "observe", "param"):
            stack = tuple(reversed(node["map_data_stack"]))
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
        vectorized_map_data_info['nodes'] = defaultdict(list)
        for name, node in nodes.items():
            if node["type"] in ("sample", "observe", "param"):
                stack = tuple(reversed(node["map_data_stack"]))
                vec_mds = list(filter(lambda x: x[2] == 'tensor', stack))
                if len(vec_mds) > 0:
                    node_batch_dim_pair = (name, vec_mds[0][3])
                    vectorized_map_data_info['nodes'][vec_mds[0][0]].append(node_batch_dim_pair)

    return vectorized_map_data_info


def save_visualization(trace, graph_output):
    """
    render graph and save to file
    :param graph_output: the graph will be saved to graph_output.pdf
    -- non-reparameterized stochastic nodes are salmon
    -- reparameterized stochastic nodes are half salmon, half grey
    -- observation nodes are green
    """
    g = graphviz.Digraph()
    for label, node in trace.nodes:
        shape = 'ellipse'
        if label in trace.stochastic_nodes and label not in trace.reparameterized_nodes:
            fillcolor = 'salmon'
        elif label in trace.reparameterized_nodes:
            fillcolor = 'lightgrey;.5:salmon'
        elif label in trace.observation_nodes:
            fillcolor = 'darkolivegreen3'
        else:
            fillcolor = 'grey'
        g.node(label, label=label, shape=shape, style='filled', fillcolor=fillcolor)

    for label1, label2 in trace.G.edges():
        g.edge(label1, label2)

    g.render(graph_output, view=False, cleanup=True)
