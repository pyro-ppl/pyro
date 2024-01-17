# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Dict, Optional

from pyro.poutine.messenger import Messenger

if TYPE_CHECKING:
    import torch

    from pyro.poutine.runtime import Message
    from pyro.poutine.trace_struct import Trace


class ReplayMessenger(Messenger):
    """
    Given a callable that contains Pyro primitive calls,
    return a callable that runs the original, reusing the values at sites in trace
    at those sites in the new trace

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    ``replay`` makes ``sample`` statements behave as if they had sampled the values
    at the corresponding sites in the trace:

        >>> old_trace = pyro.poutine.trace(model).get_trace(1.0)
        >>> replayed_model = pyro.poutine.replay(model, trace=old_trace)
        >>> bool(replayed_model(0.0) == old_trace.nodes["_RETURN"]["value"])
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param trace: a :class:`~pyro.poutine.Trace` data structure to replay against
    :param params: dict of names of param sites and constrained values
        in fn to replay against
    :returns: a stochastic function decorated with a :class:`~pyro.poutine.replay_messenger.ReplayMessenger`
    """

    def __init__(
        self,
        trace: Optional["Trace"] = None,
        params: Optional[Dict[str, "torch.Tensor"]] = None,
    ) -> None:
        """
        :param trace: a trace whose values should be reused

        Constructor.
        Stores trace in an attribute.
        """
        super().__init__()
        if trace is None and params is None:
            raise ValueError("must provide trace or params to replay against")
        self.trace = trace
        self.params = params

    def _pyro_sample(self, msg: "Message") -> None:
        """
        :param msg: current message at a trace site.

        At a sample site that appears in self.trace,
        returns the value from self.trace instead of sampling
        from the stochastic function at the site.

        At a sample site that does not appear in self.trace,
        reverts to default Messenger._pyro_sample behavior with no additional side effects.
        """
        assert msg["name"] is not None
        name = msg["name"]
        if self.trace is not None and name in self.trace:
            guide_msg = self.trace.nodes[name]
            if msg["is_observed"]:
                return None
            if guide_msg["type"] != "sample" or guide_msg["is_observed"]:
                raise RuntimeError("site {} must be sampled in trace".format(name))
            msg["done"] = True
            msg["value"] = guide_msg["value"]
            msg["infer"] = guide_msg["infer"]

    def _pyro_param(self, msg: "Message") -> None:
        name = msg["name"]
        if self.params is not None and name in self.params:
            assert hasattr(
                self.params[name], "unconstrained"
            ), "param {} must be constrained value".format(name)
            msg["done"] = True
            msg["value"] = self.params[name]
