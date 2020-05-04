# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pyro.poutine.replay_messenger import ReplayMessenger as OrigReplayMessenger
from pyro.contrib.funsor.handlers.primitives import to_data


class ReplayMessenger(OrigReplayMessenger):
    """
    This version of ReplayMessenger is almost identical to the original version,
    except that it calls to_data on the replayed funsor values.
    This may result in different unpacked shapes, but should produce correct allocations.
    """
    def _pyro_sample(self, msg):
        name = msg["name"]
        if self.trace is not None and name in self.trace:
            guide_msg = self.trace.nodes[name]
            if msg["is_observed"]:
                return None
            msg["funsor"] = {} if "funsor" not in msg else msg["funsor"]
            if guide_msg["type"] != "sample" or guide_msg["is_observed"]:
                raise RuntimeError("site {} must be sample in trace".format(name))
            # TODO make this work with sequential enumeration
            if guide_msg.get("funsor", {}).get("value", None) is not None:
                msg["value"] = to_data(guide_msg["funsor"]["value"])  # only difference is here
            else:
                msg["value"] = guide_msg["value"]
            msg["infer"] = guide_msg["infer"]
            msg["done"] = True
