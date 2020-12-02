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
    Only for internal use by ``VectorizedMarkovMessenger`` to produce
    a `step` collection for a `plate_to_step` dict.

    `plate_to_step` maps markov dim name `name` to a `step` collection.
    This function creates a step tuple for each markov var and adds it to `step`.
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


class VectorizedMarkovMessenger(GlobalNamedMessenger):
    """
    Pyro interface for ``modified_partial_sum_product``
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
        # GlobalNamedMessenger
        frame = self._saved_frames.pop() if self._saved_frames else StackFrame(
            name_to_dim=OrderedDict(), dim_to_name=OrderedDict())
        _DIM_STACK.push_global(frame)
        # IndepMessenger
        name_to_dim = OrderedDict([(self.name, DimRequest(self.dim, self.dim_type))])
        indices = to_data(self._indices, name_to_dim=name_to_dim)
        # extract the dimension allocated by to_data to match plate's current behavior
        self.dim, self.indices = -indices.dim(), indices.squeeze()
        # extra information
        self._markov_vars = set()
        self._auxiliary_to_markov = {}
        # _suffixes is used by _markov_step to create a step
        self._suffixes = []
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
        _DIM_STACK.pop_iter()
        self._saved_frames.append(_DIM_STACK.pop_global())
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

    #  def __enter__(self):
    #      super().__enter__()  # do this first to take care of globals recycling
    #      return self
    #
    #  def __exit__(self, *args):
    #      super().__exit__(*args)
    #      _markov_step(name=self.name, markov_vars=self._markov_vars, suffixes=self._suffixes)

    def _pyro_sample(self, msg):
        if type(msg["fn"]).__name__ == "_Subsample":
            return
        if not isinstance(self._suffix, int):
            frame = CondIndepStackFrame(self.name, self.dim, self.size-self.history, 0)
            msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
            BroadcastMessenger._pyro_sample(msg)
            if str(self._suffix) != str(self.indices[self.history:self.size]):
                # do not trace if not the last step in the loop
                msg["infer"]["_do_not_trace"] = True
                msg["infer"]["is_auxiliary"] = True
                msg["is_observed"] = False
                # map to markov var name
                markov = msg["name"].replace(str(self._suffix), "")
                self._auxiliary_to_markov[msg["name"]] = markov

    def _pyro_post_sample(self, msg):
        if str(self._suffix) == str(self.indices[self.history:self.size]):
            funsor_log_prob = to_funsor(msg["fn"].log_prob(msg["value"]), output=funsor.Real)
            for name in set(funsor_log_prob.inputs) & set(self._auxiliary_to_markov):
                markov = self._auxiliary_to_markov[name]
                self._markov_vars.add(markov)
