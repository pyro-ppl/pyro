# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from numbers import Number

import funsor

from pyro.contrib.funsor.handlers.named_messenger import (
    DimRequest,
    DimType,
    GlobalNamedMessenger,
    NamedMessenger,
)
from pyro.contrib.funsor.handlers.primitives import to_data, to_funsor
from pyro.distributions.util import copy_docs_from
from pyro.poutine.broadcast_messenger import BroadcastMessenger
from pyro.poutine.indep_messenger import CondIndepStackFrame
from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import effectful
from pyro.poutine.subsample_messenger import (
    SubsampleMessenger as OrigSubsampleMessenger,
)
from pyro.util import ignore_jit_warnings

funsor.set_backend("torch")


class IndepMessenger(GlobalNamedMessenger):
    """
    Vectorized plate implementation using :func:`~pyro.contrib.funsor.to_data` instead of
    :class:`~pyro.poutine.runtime._DimAllocator`.
    """

    def __init__(self, name=None, size=None, dim=None, indices=None):
        assert dim is None or dim < 0
        super().__init__()
        # without a name or dim, treat as a "vectorize" effect and allocate a non-visible dim
        self.dim_type = (
            DimType.GLOBAL if name is None and dim is None else DimType.VISIBLE
        )
        self.name = name if name is not None else funsor.interpreter.gensym("PLATE")
        self.size = size
        self.dim = dim
        if not hasattr(self, "_full_size"):
            self._full_size = size
        if indices is None:
            indices = funsor.ops.new_arange(
                funsor.tensor.get_default_prototype(), self.size
            )
        assert len(indices) == size

        self._indices = funsor.Tensor(
            indices, OrderedDict([(self.name, funsor.Bint[self.size])]), self._full_size
        )

    def __enter__(self):
        super().__enter__()  # do this first to take care of globals recycling
        name_to_dim = OrderedDict([(self.name, DimRequest(self.dim, self.dim_type))])
        indices = to_data(self._indices, name_to_dim=name_to_dim)
        # extract the dimension allocated by to_data to match plate's current behavior
        self.dim, self.indices = -indices.dim(), indices.reshape(-1)
        return self

    def _pyro_sample(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]

    def _pyro_param(self, msg):
        frame = CondIndepStackFrame(self.name, self.dim, self.size, 0)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]


@copy_docs_from(OrigSubsampleMessenger)
class SubsampleMessenger(IndepMessenger):
    def __init__(
        self,
        name=None,
        size=None,
        subsample_size=None,
        subsample=None,
        dim=None,
        use_cuda=None,
        device=None,
    ):
        size, subsample_size, indices = OrigSubsampleMessenger._subsample(
            name, size, subsample_size, subsample, use_cuda, device
        )
        self.subsample_size = subsample_size
        self._full_size = size
        self._scale = float(size) / subsample_size
        # initialize other things last
        super().__init__(name, subsample_size, dim, indices)

    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        msg["scale"] = msg["scale"] * self._scale

    def _pyro_param(self, msg):
        super()._pyro_param(msg)
        msg["scale"] = msg["scale"] * self._scale

    def _subsample_site_value(self, value, event_dim=None):
        if (
            self.dim is not None
            and event_dim is not None
            and self.subsample_size < self._full_size
        ):
            event_shape = value.shape[len(value.shape) - event_dim :]
            funsor_value = to_funsor(value, output=funsor.Reals[event_shape])
            if self.name in funsor_value.inputs:
                return to_data(funsor_value(**{self.name: self._indices}))
        return value

    def _pyro_post_param(self, msg):
        event_dim = msg["kwargs"].get("event_dim")
        new_value = self._subsample_site_value(msg["value"], event_dim)
        if new_value is not msg["value"]:
            if hasattr(msg["value"], "_pyro_unconstrained_param"):
                param = msg["value"]._pyro_unconstrained_param
            else:
                param = msg["value"].unconstrained()

            if not hasattr(param, "_pyro_subsample"):
                param._pyro_subsample = {}  # TODO is this going to persist correctly?

            param._pyro_subsample[self.dim - event_dim] = self.indices
            new_value._pyro_unconstrained_param = param
            msg["value"] = new_value

    def _pyro_post_subsample(self, msg):
        event_dim = msg["kwargs"].get("event_dim")
        msg["value"] = self._subsample_site_value(msg["value"], event_dim)


class PlateMessenger(SubsampleMessenger):
    """
    Combines new :class:`~IndepMessenger` implementation with existing
    :class:`pyro.poutine.BroadcastMessenger`. Should eventually be a drop-in
    replacement for :class:`pyro.plate`.
    """

    def __enter__(self):
        super().__enter__()
        return self.indices  # match pyro.plate behavior

    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        BroadcastMessenger._pyro_sample(msg)

    def __iter__(self):
        return iter(
            _SequentialPlateMessenger(
                self.name, self.size, self._indices.data.squeeze(), self._scale
            )
        )


class _SequentialPlateMessenger(Messenger):
    """
    Implementation of sequential plate. Should not be used directly.
    """

    def __init__(self, name, size, indices, scale):
        self.name = name
        self.size = size
        self.indices = indices
        self._scale = scale
        self._counter = 0
        super().__init__()

    def __iter__(self):
        with ignore_jit_warnings([("Iterating over a tensor", RuntimeWarning)]), self:
            self._counter = 0
            for i in self.indices:
                self._counter += 1
                yield i if isinstance(i, Number) else i.item()

    def _pyro_sample(self, msg):
        frame = CondIndepStackFrame(self.name, None, self.size, self._counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        msg["scale"] = msg["scale"] * self._scale

    def _pyro_param(self, msg):
        frame = CondIndepStackFrame(self.name, None, self.size, self._counter)
        msg["cond_indep_stack"] = (frame,) + msg["cond_indep_stack"]
        msg["scale"] = msg["scale"] * self._scale


class VectorizedMarkovMessenger(NamedMessenger):
    """
    Construct for Markov chain of variables designed for efficient elimination of Markov
    dimensions using the parallel-scan algorithm. Whenever permissible,
    :class:`~pyro.contrib.funsor.vectorized_markov` is interchangeable with
    :class:`~pyro.contrib.funsor.markov`.

    The for loop generates both :class:`int` and 1-dimensional :class:`torch.Tensor` indices:
    :code:`(0, ..., history-1, torch.arange(0, size-history), ..., torch.arange(history, size))`.
    :class:`int` indices are used to initiate the Markov chain and :class:`torch.Tensor` indices
    are used to construct vectorized transition probabilities for efficient elimination by
    the parallel-scan algorithm.

    When ``history==0`` :class:`~pyro.contrib.funsor.vectorized_markov` behaves
    similar to :class:`~pyro.contrib.funsor.plate`.

    After the for loop is run, Markov variables are identified and then the ``step``
    information is constructed and added to the trace. ``step`` informs inference algorithms
    which variables belong to a Markov chain.

    .. code-block:: py

        data = torch.ones(3, dtype=torch.float)

        def model(data, vectorized=True):

            init = pyro.param("init", lambda: torch.rand(3), constraint=constraints.simplex)
            trans = pyro.param("trans", lambda: torch.rand((3, 3)), constraint=constraints.simplex)
            locs = pyro.param("locs", lambda: torch.rand(3,))

            markov_chain = \\
                pyro.vectorized_markov(name="time", size=len(data), dim=-1) if vectorized \\
                else pyro.markov(range(len(data)))
            for i in markov_chain:
                x_curr = pyro.sample("x_{}".format(i), dist.Categorical(
                    init if isinstance(i, int) and i < 1 else trans[x_prev]),

                pyro.sample("y_{}".format(i),
                            dist.Normal(Vindex(locs)[..., x_curr], 1.),
                            obs=data[i])
                x_prev = x_curr

        #  trace.nodes["time"]["value"]
        #  frozenset({('x_0', 'x_slice(0, 2, None)', 'x_slice(1, 3, None)')})
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
        #  x_slice(1, 3, None) dist   3 1 1 1 1 2 |
        #          value 3 1 1 1 1 1 1 |
        #       log_prob 3 3 1 1 1 1 2 |
        #  y_slice(1, 3, None) dist 3 1 1 1 1 1 2 |
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

    .. warning::  This is only valid if there is only one Markov
        dimension per branch.

    :param str name: A unique name of a Markov dimension to help inference algorithm
        eliminate variables in the Markov chain.
    :param int size: Length (size) of the Markov chain.
    :param int dim: An optional dimension to use for this Markov dimension.
        If specified, ``dim`` should be negative, i.e. should index from the
        right. If not specified, ``dim`` is set to the rightmost dim that is
        left of all enclosing :class:`~pyro.contrib.funsor.plate` contexts.
    :param int history: Memory (order) of the Markov chain. Also the number
        of previous contexts visible from the current context. Defaults to 1.
        If zero, this is similar to :class:`~pyro.contrib.funsor.plate`.
    :return: Returns both :class:`int` and 1-dimensional :class:`torch.Tensor` indices:
        ``(0, ..., history-1, torch.arange(size-history), ..., torch.arange(history, size))``.
    """

    def __init__(self, name=None, size=None, dim=None, history=1):
        self.name = name
        self.size = size
        self.dim = dim
        self.history = history
        super().__init__()

    @staticmethod
    @effectful(type="markov_chain")
    def _markov_chain(name=None, markov_vars=set(), suffixes=list()):
        """
        Constructs names of markov variables in the `chain`
        from markov_vars prefixes and suffixes.

        :param str name: The name of the markov dimension.
        :param set markov_vars: Markov variable name markov_vars.
        :param list suffixes: Markov variable name suffixes.
            (`0, ..., history-1, torch.arange(0, size-history), ..., torch.arange(history, size)`)
        :return: step information
        :rtype: frozenset
        """
        chain = frozenset(
            {
                tuple("{}{}".format(var, suffix) for suffix in suffixes)
                for var in markov_vars
            }
        )
        return chain

    def __iter__(self):
        self._auxiliary_to_markov = {}
        self._markov_vars = set()
        self._suffixes = []
        for self._suffix in range(self.history):
            self._suffixes.append(self._suffix)
            yield self._suffix
        with self:
            with IndepMessenger(
                name=self.name, size=self.size - self.history, dim=self.dim
            ) as time:
                time_indices = [time.indices + i for i in range(self.history + 1)]
                time_slices = [
                    slice(i, self.size - self.history + i)
                    for i in range(self.history + 1)
                ]
                self._suffixes.extend(time_slices)
                for self._suffix, self._indices in zip(time_slices, time_indices):
                    yield self._indices
        self._markov_chain(
            name=self.name, markov_vars=self._markov_vars, suffixes=self._suffixes
        )

    def _pyro_sample(self, msg):
        if type(msg["fn"]).__name__ == "_Subsample":
            return
        BroadcastMessenger._pyro_sample(msg)
        # replace tensor suffix with a nice slice suffix
        if isinstance(self._suffix, slice):
            assert msg["name"].endswith(str(self._indices))
            msg["name"] = msg["name"][: -len(str(self._indices))] + str(self._suffix)
        if str(self._suffix) != str(self._suffixes[-1]):
            # _do_not_score: record these sites when tracing for use with replay,
            # but do not include them in ELBO computation.
            msg["infer"]["_do_not_score"] = True
            # map auxiliary var to markov var name prefix
            # assuming that site name has a format: "markov_var{}".format(_suffix)
            # is there a better way?
            markov_var = msg["name"][: -len(str(self._suffix))]
            self._auxiliary_to_markov[msg["name"]] = markov_var

    def _pyro_post_sample(self, msg):
        """
        At the last step of the for loop identify markov variables.
        """
        if type(msg["fn"]).__name__ == "_Subsample":
            return
        # if last step in the for loop
        if str(self._suffix) == str(self._suffixes[-1]):
            funsor_log_prob = (
                msg["funsor"]["log_prob"]
                if "log_prob" in msg.get("funsor", {})
                else to_funsor(msg["fn"].log_prob(msg["value"]), output=funsor.Real)
            )
            # for auxiliary sites in the log_prob
            for name in set(funsor_log_prob.inputs) & set(self._auxiliary_to_markov):
                # add markov var name prefix to self._markov_vars
                markov_var = self._auxiliary_to_markov[name]
                self._markov_vars.add(markov_var)
