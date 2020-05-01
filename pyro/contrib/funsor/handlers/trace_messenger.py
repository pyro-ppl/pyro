# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor

from pyro.poutine.subsample_messenger import _Subsample
from pyro.poutine.trace_messenger import TraceMessenger as OrigTraceMessenger

from pyro.contrib.funsor.handlers.primitives import to_funsor
from pyro.contrib.funsor.handlers.named_messenger import NamedMessenger


class TraceMessenger(OrigTraceMessenger):
    """
    This version of TraceMessenger does its packing online instead of after the fact,
    converting all distributions and values to Funsors as soon as they are available.
    """
    def _pyro_post_sample(self, msg):
        if isinstance(msg["fn"], _Subsample):
            return super()._pyro_post_sample(msg)
        if "funsor_fn" not in msg["infer"]:
            msg["infer"]["funsor_fn"] = to_funsor(msg["fn"], funsor.reals())
        if "funsor_log_prob" not in msg["infer"]:
            msg["infer"]["funsor_log_prob"] = to_funsor(msg["fn"].log_prob(msg["value"]),
                                                        funsor.reals())
        if "funsor_value" not in msg["infer"]:
            value_output = funsor.reals(*getattr(msg["fn"], "event_shape", ()))
            msg["infer"]["funsor_value"] = to_funsor(msg["value"], value_output)
        return super()._pyro_post_sample(msg)


class PackTraceMessenger(OrigTraceMessenger):
    """
    This version of TraceMessenger records information necessary to do packing after execution.
    Each sample site is annotated with a "dim_to_name" dictionary,
    which can be passed directly to funsor.to_funsor.
    """
    def _pyro_post_sample(self, msg):
        if isinstance(msg["fn"], _Subsample):
            return super()._pyro_post_sample(msg)
        msg["infer"]["dim_to_name"] = NamedMessenger._get_dim_to_name(msg["fn"].batch_shape)
        msg["infer"]["dim_to_name"].update(NamedMessenger._get_dim_to_name(
            msg["value"].shape[:len(msg["value"]).shape - len(msg["fn"].event_shape)]))
        return super()._pyro_post_sample(msg)
