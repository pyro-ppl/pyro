# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from collections import OrderedDict

from pyro.poutine import Messenger
from pyro.poutine.trace_messenger import TraceMessenger

from pyro.contrib.funsor import to_funsor, to_data


class SimpleEnumMessenger(Messenger):

    def _pyro_sample(self, msg):

        import funsor

        if msg["done"] or msg["is_observed"] or msg.get("expand", False) or \
                msg["infer"].get("enumerate") != "parallel":
            return

        msg["infer"]["funsor_fn"] = to_funsor(msg["fn"])
        size = msg["infer"]["funsor_fn"].inputs["value"].dtype
        msg["infer"]["funsor_value"] = funsor.Tensor(
            torch.arange(size),
            OrderedDict([(msg["name"], size)]),
            funsor.bint(size)
        )

        msg["value"] = to_data(msg["infer"]["funsor_value"])
        msg["done"] = True


class FunsorTraceMessenger(TraceMessenger):

    def _pyro_post_sample(self, msg):
        if "funsor_fn" not in msg["infer"]:
            msg["infer"]["funsor_fn"] = to_funsor(msg["fn"])
        if "funsor_value" not in msg["infer"]:
            msg["infer"]["funsor_value"] = to_funsor(msg["value"])
        return super()._pyro_post_sample(msg)


def simple_ve_1(model, *args):

    import funsor

    with FunsorLogJointMessenger() as tr:
        with SimpleEnumMessenger():
            model(*args)

    log_joint = sum([site["infer"]["funsor_fn"](value=site["infer"]["funsor_value"])
                     for site in tr.trace.nodes.values()])
    return log_joint.reduce(funsor.ops.logaddexp)


class FunsorLogJointMessenger(Messenger):

    def __enter__(self):
        self.log_joint = to_funsor(0.)
        return super().__enter__()

    def _pyro_post_sample(self, msg):
        import funsor
        with funsor.interpretation(funsor.terms.lazy):
            self.log_joint += msg["infer"].get("funsor_fn", to_funsor(msg["fn"]))(
                value=msg["infer"].get("funsor_value", to_funsor(msg["value"]))
            )


def simple_ve_2(model, *args):

    import funsor

    with FunsorLogJointMessenger() as tr:
        with SimpleEnumMessenger():
            model(*args)

    with funsor.interpretation(funsor.optimizer.optimize):
        return tr.log_joint.reduce(funsor.ops.logaddexp)
