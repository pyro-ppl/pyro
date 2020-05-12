# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor

from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.trace_messenger import TraceMessenger as OrigTraceMessenger

from pyro.contrib.funsor.handlers.primitives import _EmptyDist, to_data, to_funsor
from pyro.contrib.funsor.handlers.runtime import _DIM_STACK


class TraceMessenger(OrigTraceMessenger):
    """
    Setting ``pack_online=True`` packs online instead of after the fact,
    converting all distributions and values to Funsors as soon as they are available.

    Setting ``pack_online=False`` computes information necessary to do packing after execution.
    Each sample site is annotated with a "dim_to_name" dictionary,
    which can be passed directly to funsor.to_funsor.
    """
    def _pyro_post_sample(self, msg):
        if msg["name"] in self.trace and isinstance(msg["fn"], _EmptyDist):
            return
        if "funsor" not in msg:
            msg["funsor"] = {}
        if isinstance(msg["fn"], _Subsample):
            return super()._pyro_post_sample(msg)
        if "fn" not in msg["funsor"]:
            fn_masked = msg["fn"].mask(msg["mask"]) if msg["mask"] is not None else msg["fn"]
            msg["funsor"]["fn"] = to_funsor(fn_masked, funsor.reals())(value=msg["name"])
        if "value" not in msg["funsor"]:
            # value_output = funsor.reals(*getattr(msg["fn"], "event_shape", ()))
            msg["funsor"]["value"] = to_funsor(msg["value"], msg["funsor"]["fn"].inputs[msg["name"]])
        if "log_prob" not in msg["funsor"]:
            fn_masked = msg["fn"].mask(msg["mask"]) if msg["mask"] is not None else msg["fn"]
            msg["funsor"]["log_prob"] = to_funsor(fn_masked.log_prob(msg["value"]), output=funsor.reals())
            # TODO support this pattern which uses funsor directly - blocked by casting issues in funsor
            # msg["funsor"]["log_prob"] = msg["funsor"]["fn"](**{msg["name"]: msg["funsor"]["value"]})
        if msg["scale"] is not None and "scale" not in msg["funsor"]:
            msg["funsor"]["scale"] = to_funsor(msg["scale"], output=funsor.reals())
        return super()._pyro_post_sample(msg)
