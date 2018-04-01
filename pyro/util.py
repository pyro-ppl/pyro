from __future__ import absolute_import, division, print_function

import functools
import numbers
import random
import warnings
from collections import defaultdict

import graphviz
import torch
from six.moves import zip_longest

from pyro.params import _PYRO_PARAM_STORE
from pyro.poutine.poutine import _PYRO_STACK
from pyro.poutine.util import site_is_subsample


def validate_message(msg):
    """
    Asserts that the message has a valid format.
    :returns: None
    """
    assert msg["type"] in ("sample", "param"), \
        "{} is an invalid site type, how did that get there?".format(msg["type"])


def default_process_message(msg):
    """
    Default method for processing messages in inference.
    :param msg: a message to be processed
    :returns: None
    """
    validate_message(msg)
    if msg["type"] == "sample":
        fn, args, kwargs = \
            msg["fn"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg

        if msg["is_observed"]:
            assert msg["value"] is not None
            val = msg["value"]
        else:
            val = fn(*args, **kwargs)

        # after fn has been called, update msg to prevent it from being called again.
        msg["done"] = True
        msg["value"] = val
    elif msg["type"] == "param":
        name, args, kwargs = \
            msg["name"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg

        ret = _PYRO_PARAM_STORE.get_param(name, *args, **kwargs)

        # after the param store has been queried, update msg["done"]
        # to prevent it from being queried again.
        msg["done"] = True
        msg["value"] = ret
    else:
        assert False
    return None


def am_i_wrapped():
    """
    Checks whether the current computation is wrapped in a poutine.
    :returns: bool
    """
    return len(_PYRO_STACK) > 0


def detach_iterable(iterable):
    if torch.is_tensor(iterable):
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
    Sets seeds of torch and torch.cuda (if available).
    :param int rng_seed: The seed value.
    """
    torch.manual_seed(rng_seed)
    random.seed(rng_seed)
    try:
        import numpy as np

        np.random.seed(rng_seed)
    except ImportError:
        pass


def is_nan(x):
    """
    A convenient function to check if a Tensor contains all nan; also works with numbers
    """
    if isinstance(x, numbers.Number):
        return x != x
    return (x != x).all()


def is_inf(x):
    """
    A convenient function to check if a Tensor contains all inf; also works with numbers
    """
    if isinstance(x, numbers.Number):
        return x == float('inf')
    return (x == float('inf')).all()


def log_sum_exp(tensor):
    max_val = tensor.max(dim=-1)[0]
    return max_val + (tensor - max_val.unsqueeze(-1)).exp().sum(dim=-1).log()


def zero_grads(tensors):
    """
    Sets gradients of list of Variables to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            p.grad = p.grad.new(p.shape).zero_()


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
    stack = _PYRO_STACK
    # TODO check at runtime if stack is valid

    # msg is used to pass information up and down the stack
    msg = initial_msg

    counter = 0
    # go until time to stop?
    for frame in stack:
        validate_message(msg)

        counter = counter + 1

        frame._process_message(msg)

        if msg["stop"]:
            break

    default_process_message(msg)

    for frame in reversed(stack[0:counter]):
        frame._postprocess_message(msg)

    cont = msg["continuation"]
    if cont is not None:
        cont(msg)

    return None


def save_visualization(trace, graph_output):
    """
    :param pyro.poutine.Trace trace: a trace to be visualized
    :param graph_output: the graph will be saved to graph_output.pdf
    :type graph_output: str

    Take a trace generated by poutine.trace with `graph_type='dense'` and render
    the graph with the output saved to file.

    - non-reparameterized stochastic nodes are salmon
    - reparameterized stochastic nodes are half salmon, half grey
    - observation nodes are green

    Example:

    trace = pyro.poutine.trace(model, graph_type="dense").get_trace()
    save_visualization(trace, 'output')
    """
    g = graphviz.Digraph()

    for label, node in trace.nodes.items():
        if site_is_subsample(node):
            continue
        shape = 'ellipse'
        if label in trace.stochastic_nodes and label not in trace.reparameterized_nodes:
            fillcolor = 'salmon'
        elif label in trace.reparameterized_nodes:
            fillcolor = 'lightgrey;.5:salmon'
        elif label in trace.observation_nodes:
            fillcolor = 'darkolivegreen3'
        else:
            # only visualize RVs
            continue
        g.node(label, label=label, shape=shape, style='filled', fillcolor=fillcolor)

    for label1, label2 in trace.edges:
        if site_is_subsample(trace.nodes[label1]):
            continue
        if site_is_subsample(trace.nodes[label2]):
            continue
        g.edge(label1, label2)

    g.render(graph_output, view=False, cleanup=True)


def check_traces_match(trace1, trace2):
    """
    :param pyro.poutine.Trace trace1: Trace object of the model
    :param pyro.poutine.Trace trace2: Trace object of the guide
    :raises: RuntimeWarning, ValueError

    Checks that (1) there is a bijection between the samples in the two traces
    and (2) at each sample site two traces agree on sample shape.
    """
    # Check ordinary sample sites.
    vars1 = set(name for name, site in trace1.nodes.items() if site["type"] == "sample")
    vars2 = set(name for name, site in trace2.nodes.items() if site["type"] == "sample")
    if vars1 != vars2:
        warnings.warn("Model vars changed: {} vs {}".format(vars1, vars2))

    # Check shapes agree.
    for name in vars1:
        site1 = trace1.nodes[name]
        site2 = trace2.nodes[name]
        if hasattr(site1["fn"], "shape") and hasattr(site2["fn"], "shape"):
            shape1 = site1["fn"].shape(*site1["args"], **site1["kwargs"])
            shape2 = site2["fn"].shape(*site2["args"], **site2["kwargs"])
            if shape1 != shape2:
                raise ValueError("Site dims disagree at site '{}': {} vs {}".format(name, shape1, shape2))


def check_model_guide_match(model_trace, guide_trace, max_iarange_nesting=float('inf')):
    """
    :param pyro.poutine.Trace model_trace: Trace object of the model
    :param pyro.poutine.Trace guide_trace: Trace object of the guide
    :raises: RuntimeWarning, ValueError

    Checks that (1) there is a bijection between the samples in the guide
    and the samples in the model, (2) each `iarange` statement in the guide
    also appears in the model, (3) at each sample site that appears in both
    the model and guide, the model and guide agree on sample shape.
    """
    # Check ordinary sample sites.
    model_vars = set(name for name, site in model_trace.nodes.items()
                     if site["type"] == "sample" and not site["is_observed"]
                     if type(site["fn"]).__name__ != "_Subsample")
    guide_vars = set(name for name, site in guide_trace.nodes.items()
                     if site["type"] == "sample"
                     if type(site["fn"]).__name__ != "_Subsample")
    if not (guide_vars <= model_vars):
        warnings.warn("Found vars in guide but not model: {}".format(guide_vars - model_vars))
    if not (model_vars <= guide_vars):
        warnings.warn("Found vars in model but not guide: {}".format(model_vars - guide_vars))

    # Check shapes agree.
    for name in model_vars & guide_vars:
        model_site = model_trace.nodes[name]
        guide_site = guide_trace.nodes[name]

        if hasattr(model_site["fn"], "event_dim") and hasattr(guide_site["fn"], "event_dim"):
            if model_site["fn"].event_dim != guide_site["fn"].event_dim:
                raise ValueError("Model and guide event_dims disagree at site '{}': {} vs {}".format(
                    name, model_site["fn"].event_dim, guide_site["fn"].event_dim))

        if hasattr(model_site["fn"], "shape") and hasattr(guide_site["fn"], "shape"):
            model_shape = model_site["fn"].shape(*model_site["args"], **model_site["kwargs"])
            guide_shape = guide_site["fn"].shape(*guide_site["args"], **guide_site["kwargs"])
            if model_shape == guide_shape:
                continue

            # Allow broadcasting outside of max_iarange_nesting.
            if len(model_shape) > max_iarange_nesting:
                model_shape = model_shape[len(model_shape) - max_iarange_nesting:]
            if len(guide_shape) > max_iarange_nesting:
                guide_shape = guide_shape[len(guide_shape) - max_iarange_nesting:]
            if model_shape == guide_shape:
                continue
            for model_size, guide_size in zip_longest(reversed(model_shape), reversed(guide_shape), fillvalue=1):
                if model_size != guide_size:
                    raise ValueError("Model and guide shapes disagree at site '{}': {} vs {}".format(
                        name, model_shape, guide_shape))

    # Check subsample sites introduced by iarange.
    model_vars = set(name for name, site in model_trace.nodes.items()
                     if site["type"] == "sample" and not site["is_observed"]
                     if type(site["fn"]).__name__ == "_Subsample")
    guide_vars = set(name for name, site in guide_trace.nodes.items()
                     if site["type"] == "sample"
                     if type(site["fn"]).__name__ == "_Subsample")
    if not (guide_vars <= model_vars):
        warnings.warn("Found iarange statements in guide but not model: {}".format(guide_vars - model_vars))


def check_site_shape(site, max_iarange_nesting):
    actual_shape = list(site["log_prob"].shape)

    # Compute expected shape.
    expected_shape = []
    for f in site["cond_indep_stack"]:
        if f.dim is not None:
            # Use the specified iarange dimension, which counts from the right.
            assert f.dim < 0
            if len(expected_shape) < -f.dim:
                expected_shape = [None] * (-f.dim - len(expected_shape)) + expected_shape
            if expected_shape[f.dim] is not None:
                raise ValueError('\n  '.join([
                    'at site "{}" within iarange("", dim={}), dim collision'.format(site["name"], f.name, f.dim),
                    'Try setting dim arg in other iaranges.']))
            expected_shape[f.dim] = f.size
    expected_shape = [1 if e is None else e for e in expected_shape]

    # Check for iarange stack overflow.
    if len(expected_shape) > max_iarange_nesting:
        raise ValueError('\n  '.join([
            'at site "{}", iarange stack overflow'.format(site["name"]),
            'Try increasing max_iarange_nesting to at least {}'.format(len(expected_shape))]))

    # Ignore dimensions left of max_iarange_nesting.
    if max_iarange_nesting < len(actual_shape):
        actual_shape = actual_shape[len(actual_shape) - max_iarange_nesting:]

    # Check for incorrect iarange placement on the right of max_iarange_nesting.
    for actual_size, expected_size in zip_longest(reversed(actual_shape), reversed(expected_shape), fillvalue=1):
        if expected_size != -1 and expected_size != actual_size:
            raise ValueError('\n  '.join([
                'at site "{}", invalid log_prob shape'.format(site["name"]),
                'Expected {}, actual {}'.format(expected_shape, actual_shape),
                'Try one of the following fixes:',
                '- enclose the batched tensor in a with iarange(...): context',
                '- .reshape(extra_event_dims=...) the distribution being sampled',
                '- .permute() data dimensions']))

    # TODO Check parallel dimensions on the left of max_iarange_nesting.


def _are_independent(counters1, counters2):
    for name, counter1 in counters1.items():
        if name in counters2:
            if counters2[name] != counter1:
                return True
    return False


def check_traceenum_requirements(model_trace, guide_trace):
    """
    Warn if user could easily rewrite the model or guide in a way that would
    clearly avoid invalid dependencies on enumerated variables.

    :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO` enumerates over
    synchronized products rather than full cartesian products. Therefore models
    must ensure that no variable outside of an iarange depends on an enumerated
    variable inside that iarange. Since full dependency checking is impossible,
    this function aims to warn only in cases where models can be easily
    rewitten to be obviously correct.
    """
    enumerated_sites = set(name for name, site in guide_trace.nodes.items()
                           if site["type"] == "sample" and site["infer"].get("enumerate"))
    for role, trace in [('model', model_trace), ('guide', guide_trace)]:
        irange_counters = {}
        enumerated_contexts = defaultdict(set)
        for name, site in trace.nodes.items():
            if site["type"] != "sample":
                continue
            irange_counter = {f.name: f.counter for f in site["cond_indep_stack"] if not f.vectorized}
            context = frozenset(f for f in site["cond_indep_stack"] if f.vectorized)

            # Check that sites outside each independence context precede enumerated sites inside that context.
            for enumerated_context, names in enumerated_contexts.items():
                if not (context < enumerated_context):
                    continue
                names = sorted(n for n in names if not _are_independent(irange_counter, irange_counters[n]))
                if not names:
                    continue
                diff = sorted(f.name for f in enumerated_context - context)
                warnings.warn('\n  '.join([
                    'at {} site "{}", possibly invalid dependency.'.format(role, name),
                    'Expected site "{}" to precede sites "{}"'.format(name, '", "'.join(sorted(names))),
                    'to avoid breaking independence of iaranges "{}"'.format('", "'.join(diff)),
                ]), RuntimeWarning)

            irange_counters[name] = irange_counter
            if name in enumerated_sites:
                enumerated_contexts[context].add(name)


def deep_getattr(obj, name):
    """
    Python getattr() for arbitrarily deep attributes
    Throws an AttributeError if bad attribute
    """
    return functools.reduce(getattr, name.split("."), obj)
