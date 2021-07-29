# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor
import torch

from pyro.contrib.funsor.handlers.primitives import to_funsor
from pyro.contrib.funsor.handlers.runtime import _DIM_STACK
from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.trace_messenger import TraceMessenger as OrigTraceMessenger


def _mask_fn(fn, mask):
    if mask is not None:
        mask = torch.as_tensor(mask)  # FIXME support mask=False in funsor.distributions
        fn = fn.mask(mask)
    return fn


class TraceMessenger(OrigTraceMessenger):
    """
    Setting ``pack_online=True`` packs online instead of after the fact,
    converting all distributions and values to Funsors as soon as they are available.

    Setting ``pack_online=False`` computes information necessary to do packing after execution.
    Each sample site is annotated with a ``dim_to_name`` dictionary,
    which can be passed directly to :func:`~pyro.contrib.funsor.to_funsor`.
    """

    def __init__(self, graph_type=None, param_only=None, pack_online=True):
        super().__init__(graph_type=graph_type, param_only=param_only)
        self.pack_online = True if pack_online is None else pack_online

    def _pyro_post_sample(self, msg):
        if msg["name"] in self.trace:
            return
        if "funsor" not in msg:
            msg["funsor"] = {}
        if isinstance(msg["fn"], _Subsample):
            return super()._pyro_post_sample(msg)
        if self.pack_online:
            if "fn" not in msg["funsor"]:
                fn_masked = _mask_fn(msg["fn"], msg["mask"])
                msg["funsor"]["fn"] = to_funsor(fn_masked, funsor.Real)(
                    value=msg["name"]
                )
            if "value" not in msg["funsor"]:
                # value_output = funsor.Reals[getattr(msg["fn"], "event_shape", ())]
                msg["funsor"]["value"] = to_funsor(
                    msg["value"], msg["funsor"]["fn"].inputs[msg["name"]]
                )
            if (
                "log_prob" not in msg["funsor"]
                and not msg["infer"].get("_do_not_trace")
                and not msg["infer"].get("_do_not_score", False)
            ):
                # optimization: don't perform this tensor op unless we have to
                fn_masked = _mask_fn(msg["fn"], msg["mask"])
                msg["funsor"]["log_prob"] = to_funsor(
                    fn_masked.log_prob(msg["value"]), output=funsor.Real
                )
                # TODO support this pattern which uses funsor directly - blocked by casting issues
                # msg["funsor"]["log_prob"] = msg["funsor"]["fn"](**{msg["name"]: msg["funsor"]["value"]})
            if msg["scale"] is not None and "scale" not in msg["funsor"]:
                msg["funsor"]["scale"] = to_funsor(msg["scale"], output=funsor.Real)
        else:
            # this logic has the same side effect on the _DIM_STACK as the above,
            # but does not perform any tensor or funsor operations.
            msg["funsor"]["dim_to_name"] = _DIM_STACK.names_from_batch_shape(
                msg["fn"].batch_shape
            )
            msg["funsor"]["dim_to_name"].update(
                _DIM_STACK.names_from_batch_shape(
                    msg["value"].shape[
                        : len(msg["value"]).shape - len(msg["fn"].event_shape)
                    ]
                )
            )
        return super()._pyro_post_sample(msg)

    def _pyro_post_markov_chain(self, msg):
        # add step information to the trace
        self.trace.add_node(msg["name"], **msg.copy())
