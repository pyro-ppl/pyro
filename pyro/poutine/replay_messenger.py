# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample

if TYPE_CHECKING:
    from pyro.poutine.runtime import Message
    from pyro.poutine.trace_struct import Trace


def _subsample_values_equal(a: Any, b: Any) -> bool:
    """Compare two subsample index values for equality."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return False
    return bool(a == b)


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
            # Warn when replaying a subsample site whose explicit value differs
            # from the guide's independently drawn subsample. This happens when a
            # model passes an explicit ``subsample=idx`` to ``pyro.plate`` but is
            # composed with a guide that draws its own subsample (e.g. an
            # ``AutoGuide`` built without ``create_plates=``). Silently overriding
            # the model's index decouples the model's and guide's minibatches.
            # See https://github.com/pyro-ppl/pyro/issues/3468
            if (
                msg["value"] is not None
                and guide_msg["value"] is not None
                and site_is_subsample(msg)
                and not _subsample_values_equal(msg["value"], guide_msg["value"])
            ):
                warnings.warn(
                    "Replaying the subsample site '{}' with a value that differs "
                    "from the model's explicit subsample. If the model passes an "
                    "explicit ``subsample=idx`` to ``pyro.plate``, use a guide that "
                    "reuses the same subsample (e.g. pass ``create_plates=`` to "
                    "``AutoGuide``) so the model's and guide's minibatches stay "
                    "aligned.".format(name),
                    stacklevel=2,
                )
            msg["done"] = True
            msg["value"] = guide_msg["value"]
            msg["infer"] = guide_msg["infer"]

    def _pyro_param(self, msg: "Message") -> None:
        name = msg["name"]
        if self.params is not None and name in self.params:
            assert hasattr(self.params[name], "unconstrained"), (
                "param {} must be constrained value".format(name)
            )
            msg["done"] = True
            msg["value"] = self.params[name]
