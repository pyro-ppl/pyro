# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from contextlib import ExitStack

import funsor

from pyro.poutine.reentrant_messenger import ReentrantMessenger
from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.runtime import effectful

from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.contrib.funsor.handlers.runtime import _DIM_STACK, DimRequest, DimType, StackFrame


@effectful(type="markov_step")
def _markov_step(name, markov_vars, suffixes):
    """
    Constructs `step` information for markov_vars using suffixes.
    Only for internal use by ``VectorizedMarkovMessenger`` to produce
    a `step` that informs inference algorithms that use efficient
    elimination of markov dims.

    :param str name: The name of the markov dimension.
    :param set markov_vars: Markov variable name prefixes.
    :param list suffixes: Markov variable name suffixes.
        (`0, ..., history-1, torch.arange(0, size-history), ..., torch.arange(history, size)`)
    :return: step information
    :rtype: frozenset
    """
    step = frozenset()
    for var in markov_vars:
        step |= frozenset({tuple("{}{}".format(var, suffix) for suffix in suffixes)})
    return step


class NamedMessenger(ReentrantMessenger):
    """
    Base effect handler class for the :func:~`pyro.contrib.funsor.to_funsor`
    and :func:~`pyro.contrib.funsor.to_data` primitives.
    Any effect handlers that invoke these primitives internally or wrap
    code that does should inherit from ``NamedMessenger``.

    This design ensures that the global name-dim mapping is reset upon handler exit
    rather than potentially persisting until the entire program terminates.
    """
    def __init__(self, first_available_dim=None):
        assert first_available_dim is None or first_available_dim < 0, first_available_dim
        self.first_available_dim = first_available_dim
        self._saved_dims = set()
        return super().__init__()

    def __enter__(self):
        if self._ref_count == 0:
            if self.first_available_dim is not None:
                self._prev_first_dim = _DIM_STACK.set_first_available_dim(self.first_available_dim)
            if _DIM_STACK.outermost is None:
                _DIM_STACK.outermost = self
            for name, dim in self._saved_dims:
                _DIM_STACK.global_frame[name] = dim
            self._saved_dims = set()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self._ref_count == 1:
            if self.first_available_dim is not None:
                _DIM_STACK.set_first_available_dim(self._prev_first_dim)
            if _DIM_STACK.outermost is self:
                _DIM_STACK.outermost = None
                _DIM_STACK.set_first_available_dim(_DIM_STACK.DEFAULT_FIRST_DIM)
                self._saved_dims |= set(_DIM_STACK.global_frame.name_to_dim.items())
            for name, dim in self._saved_dims:
                del _DIM_STACK.global_frame[name]
        return super().__exit__(*args, **kwargs)

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_data(msg):

        funsor_value, = msg["args"]
        name_to_dim = msg["kwargs"].setdefault("name_to_dim", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        batch_names = tuple(funsor_value.inputs.keys())

        # interpret all names/dims as requests since we only run this function once
        name_to_dim_request = name_to_dim.copy()
        for name in batch_names:
            dim = name_to_dim.get(name, None)
            name_to_dim_request[name] = dim if isinstance(dim, DimRequest) else DimRequest(dim, dim_type)

        # request and update name_to_dim in-place
        # name_to_dim.update(_DIM_STACK.allocate_name_to_dim(name_to_dim_request))
        name_to_dim.update(_DIM_STACK.allocate(name_to_dim_request))

        msg["stop"] = True  # only need to run this once per to_data call

    @staticmethod  # only depends on the global _DIM_STACK state, not self
    def _pyro_to_funsor(msg):

        if len(msg["args"]) == 2:
            raw_value, output = msg["args"]
        else:
            raw_value = msg["args"][0]
            output = msg["kwargs"].setdefault("output", None)
        dim_to_name = msg["kwargs"].setdefault("dim_to_name", OrderedDict())
        dim_type = msg["kwargs"].setdefault("dim_type", DimType.LOCAL)

        event_dim = len(output.shape) if output else 0
        try:
            batch_shape = raw_value.batch_shape  # TODO make make this more robust
        except AttributeError:
            full_shape = getattr(raw_value, "shape", ())
            batch_shape = full_shape[:len(full_shape) - event_dim]

        batch_dims = tuple(dim for dim in range(-len(batch_shape), 0) if batch_shape[dim] > 1)

        # interpret all names/dims as requests since we only run this function once
        dim_to_name_request = dim_to_name.copy()
        for dim in batch_dims:
            name = dim_to_name.get(dim, None)
            dim_to_name_request[dim] = name if isinstance(name, DimRequest) else DimRequest(name, dim_type)

        # request and update dim_to_name in-place
        dim_to_name.update(_DIM_STACK.allocate(dim_to_name_request))

        msg["stop"] = True  # only need to run this once per to_funsor call


class MarkovMessenger(NamedMessenger):
    """
    Handler for converting to/from funsors consistent with Pyro's positional batch dimensions.

    :param int history: The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    """
    def __init__(self, history=1, keep=False):
        self.history = history
        self.keep = keep
        self._iterable = None
        self._saved_frames = []
        super().__init__()

    def __call__(self, fn):
        if fn is not None and not callable(fn):
            self._iterable = fn
            return self
        return super().__call__(fn)

    def __iter__(self):
        assert self._iterable is not None
        _DIM_STACK.push_iter(_DIM_STACK.local_frame)
        with ExitStack() as stack:
            for value in self._iterable:
                stack.enter_context(self)
                yield value
        _DIM_STACK.pop_iter()

    def __enter__(self):
        if self.keep and self._saved_frames:
            frame = self._saved_frames.pop()
        else:
            frame = StackFrame(
                name_to_dim=OrderedDict(), dim_to_name=OrderedDict(),
                history=self.history, keep=self.keep,
            )

        _DIM_STACK.push_local(frame)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self.keep:
            self._saved_frames.append(_DIM_STACK.pop_local())
        else:
            _DIM_STACK.pop_local()
        return super().__exit__(*args, **kwargs)


class GlobalNamedMessenger(NamedMessenger):
    """
    Base class for any new effect handlers that use the
    :func:~`pyro.contrib.funsor.to_funsor` and :func:~`pyro.contrib.funsor.to_data` primitives
    to allocate `DimType.GLOBAL` or `DimType.VISIBLE` dimensions.

    Serves as a manual "scope" for dimensions that should not be recycled by :class:~`MarkovMessenger`:
    global dimensions will be considered active until the innermost ``GlobalNamedMessenger``
    under which they were initially allocated exits.
    """
    def __init__(self, first_available_dim=None):
        self._saved_frames = []
        super().__init__(first_available_dim=first_available_dim)

    def __enter__(self):
        frame = self._saved_frames.pop() if self._saved_frames else StackFrame(
            name_to_dim=OrderedDict(), dim_to_name=OrderedDict())
        _DIM_STACK.push_global(frame)
        return super().__enter__()

    def __exit__(self, *args):
        self._saved_frames.append(_DIM_STACK.pop_global())
        return super().__exit__(*args)


class VectorizedMarkovMessenger(NamedMessenger):
    """
    Construct for Markov chain of variables designed for efficient elimination of Markov
    dimensions using the parallel-scan algorithm. Whenever permissible, `pyro.vectorized_markov`
    is interchangeable with `pyro.markov`.

    The for loop generates both `int` and 1-dimensional :class:`torch.Tensor` indices:
    (`0, ..., history-1, torch.arange(0, size), ..., torch.arange(history, size-history)`).
    `int` indices are used to initiate the Markov chain and :class:`torch.Tensor` indices
    are used to construct vectorized transition probabilities for efficient elimination by
    the parallel-scan algorithm.

    When `history==0` `pyro.vectorized_markov` behaves like a `pyro.plate`.
    After the for loop is run, Markov variables are identified and then the `step`
    information is constructed and added to the trace. `step` informs inference algorithms
    which variables belong to a Markov chain.

    .. code-block:: py

        data = torch.ones(3, dtype=torch.float)

        def model(data, vectorized=True):

            init = pyro.param("init", lambda: torch.rand(3), constraint=constraints.simplex)
            trans = pyro.param("trans", lambda: torch.rand((3, 3)), constraint=constraints.simplex)
            locs = pyro.param("locs", lambda: torch.rand(3,))

            markov_chain = \
                pyro.vectorized_markov(name="time", size=len(data), dim=-1) if vectorized \
                else pyro.markov(range(len(data)))
            for i in markov_chain:
                x_curr = pyro.sample("x_{}".format(i), dist.Categorical(
                    init if isinstance(i, int) and i < 1 else trans[x_prev]),

                pyro.sample("y_{}".format(i),
                            dist.Normal(Vindex(locs)[..., x_curr], 1.),
                            obs=data[i])
                x_prev = x_curr

        #  trace.nodes["time"]["infer"]["step"]
        #  frozenset({('x_0', 'x_tensor([0, 1])', 'x_tensor([1, 2])')})
        #
        #  pyro.vectorized_markov trace
        #  ...
        #  Sample Sites:
        #      locs dist               | 3
        #          value               | 3
        #       log_prob               |
        #       x_0 dist               |
        #          value     3 1 1 1 1 |
        #       log_prob     3 1 1 1 1 |
        #       y_0 dist     3 1 1 1 1 |
        #          value               |
        #       log_prob     3 1 1 1 1 |
        #  x_tensor([1, 2]) dist   3 1 1 1 1 2 |
        #          value 3 1 1 1 1 1 1 |
        #       log_prob 3 3 1 1 1 1 2 |
        #  y_tensor([1, 2]) dist 3 1 1 1 1 1 2 |
        #          value             2 |
        #       log_prob 3 1 1 1 1 1 2 |
        #
        #  pyro.markov trace
        #  ...
        #  Sample Sites:
        #      locs dist             | 3
        #          value             | 3
        #       log_prob             |
        #       x_0 dist             |
        #          value   3 1 1 1 1 |
        #       log_prob   3 1 1 1 1 |
        #       y_0 dist   3 1 1 1 1 |
        #          value             |
        #       log_prob   3 1 1 1 1 |
        #       x_1 dist   3 1 1 1 1 |
        #          value 3 1 1 1 1 1 |
        #       log_prob 3 3 1 1 1 1 |
        #       y_1 dist 3 1 1 1 1 1 |
        #          value             |
        #       log_prob 3 1 1 1 1 1 |
        #       x_2 dist 3 1 1 1 1 1 |
        #          value   3 1 1 1 1 |
        #       log_prob 3 3 1 1 1 1 |
        #       y_2 dist   3 1 1 1 1 |
        #          value             |
        #       log_prob   3 1 1 1 1 |

    .. warning::  This is only correct if there is only one Markov
        dimension per branch.

    :param str name: A unique name of a Markov dimension to help inference algorithm
        eliminate variables in the Markov chain.
    :param int size: Length (size) of the Markov chain.
    :param int dim: An optional dimension to use for this independence index.
        If specified, ``dim`` should be negative, i.e. should index from the
        right. If not specified, ``dim`` is set to the rightmost dim that is
        left of all enclosing ``plate`` contexts.
    :param int history: Memory (order) of the Markov chain. The number of previous contexts visible from the
        current context. Defaults to 1. If zero, this is similar to
        :class:`pyro.plate`.
    :param bool keep: If true, frames are replayable. This is important
        when branching: if ``keep=True``, neighboring branches at the same
        level can depend on each other; if ``keep=False``, neighboring branches
        are independent (conditioned on their shared ancestors).
    :return: First, returns `history` number of `int` indices that initialize the Markov chain (`0,..,history-1`).
        Then, returns `history+1` number of 1-dimensional :class:`torch.Tensor` indices for each `step`
        (`torch.arange(0, size-history),...,torch.arange(history, size)`).
    """
    def __init__(self, name=None, size=None, dim=None, history=1, keep=False):
        self.keep = keep
        self.history = history
        self.dim_type = DimType.GLOBAL if name is None and dim is None else DimType.VISIBLE
        self.name = name if name is not None else funsor.interpreter.gensym("MARKOV")
        self.size = size
        self.dim = dim
        indices = funsor.ops.new_arange(funsor.tensor.get_default_prototype(), self.size)
        assert len(indices) == self.size

        self._indices = funsor.Tensor(
            indices, OrderedDict([(self.name, funsor.Bint[self.size])]), self.size
        )
        self._saved_frames = []
        super().__init__()

    def __iter__(self):
        # VectorizedMarkovMessenger
        self._auxiliary_to_markov = {}
        self._markov_vars = set()
        self._suffixes = []
        # GlobalNamedMessenger
        frame = self._saved_frames.pop() if self._saved_frames else StackFrame(
            name_to_dim=OrderedDict(), dim_to_name=OrderedDict())
        _DIM_STACK.push_global(frame)
        # IndepMessenger
        name_to_dim = OrderedDict([(self.name, DimRequest(self.dim, self.dim_type))])
        indices = to_data(self._indices, name_to_dim=name_to_dim)
        # extract the dimension allocated by to_data to match plate's current behavior
        self.dim, self.indices = -indices.dim(), indices.squeeze()
        # MarkovMessenger
        _DIM_STACK.push_iter(_DIM_STACK.local_frame)
        with ExitStack() as stack:
            for i in range(2*self.history+1):
                stack.enter_context(self)
                if i < self.history:
                    # init factors
                    self._suffix = i
                else:
                    # vectorized trans factors
                    i -= self.history
                    self._suffix = self.indices[i:self.size+i-self.history]
                self._suffixes.append(self._suffix)
                yield self._suffix
        # MarkovMessenger
        _DIM_STACK.pop_iter()
        # GlobalNamedMessenger
        self._saved_frames.append(_DIM_STACK.pop_global())
        # VectorizedMarkovMessenger
        _markov_step(name=self.name, markov_vars=self._markov_vars, suffixes=self._suffixes)

    def __enter__(self):
        if self.keep and self._saved_frames:
            frame = self._saved_frames.pop()
        else:
            frame = StackFrame(
                name_to_dim=OrderedDict(), dim_to_name=OrderedDict(),
                history=self.history, keep=self.keep,
            )

        _DIM_STACK.push_local(frame)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        if self.keep:
            self._saved_frames.append(_DIM_STACK.pop_local())
        else:
            _DIM_STACK.pop_local()
        return super().__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if type(msg["fn"]).__name__ == "_Subsample":
            return
        if not isinstance(self._suffix, int):
            # use cond indep stack for vectorized indices
            frame = CondIndepStackFrame(self.name, self.dim, self.size-self.history, 0)
            msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
            BroadcastMessenger._pyro_sample(msg)
            if str(self._suffix) != str(self.indices[self.history:self.size]):
                # auxiliary vars
                # do not trace if not the last step in the loop
                msg["infer"]["_do_not_trace"] = True
                msg["infer"]["is_auxiliary"] = True
                msg["is_observed"] = False
                # map auxiliary var to markov var name prefix
                markov = msg["name"].replace(str(self._suffix), "")
                self._auxiliary_to_markov[msg["name"]] = markov

    def _pyro_post_sample(self, msg):
        """
        At the last step of the for loop identify markov variables.
        """
        # if last step in the for loop
        if str(self._suffix) == str(self.indices[self.history:self.size]):
            funsor_log_prob = to_funsor(msg["fn"].log_prob(msg["value"]), output=funsor.Real)
            # for auxiliary sites in the log_prob
            for name in set(funsor_log_prob.inputs) & set(self._auxiliary_to_markov):
                # add markov var name prefix to self._markov_vars
                markov = self._auxiliary_to_markov[name]
                self._markov_vars.add(markov)
