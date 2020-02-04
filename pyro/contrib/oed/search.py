# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import queue
from pyro.infer.abstract_infer import TracePosterior
import pyro.poutine as poutine

###################################
# Search borrowed from RSA example
###################################


class Search(TracePosterior):
    """
    Exact inference by enumerating over all possible executions
    """
    def __init__(self, model, max_tries=int(1e6), **kwargs):
        self.model = model
        self.max_tries = max_tries
        super().__init__(**kwargs)

    def _traces(self, *args, **kwargs):
        q = queue.Queue()
        q.put(poutine.Trace())
        p = poutine.trace(
            poutine.queue(self.model, queue=q, max_tries=self.max_tries))
        while not q.empty():
            tr = p.get_trace(*args, **kwargs)
            yield tr, tr.log_prob_sum()
