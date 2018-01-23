from __future__ import absolute_import, division, print_function

from collections import namedtuple

from .poutine import Messenger

CondIndepStackFrame = namedtuple("CondIndepStackFrame", ["name", "counter", "vectorized"])


class IndepMessenger(Messenger):
    """
    This messenger keeps track of stack of independence information declared by
    nested ``irange`` and ``iarange`` contexts. This information is stored in
    a ``cond_indep_stack`` at each sample/observe site for consumption by
    ``TracePoutine``.
    """
    def __init__(self, name, vectorized):
        """
        Constructor: basically default, but store a counter to keep track of
        which ``irange`` branch we're in.
        """
        super(IndepMessenger, self).__init__()
        self.name = name
        self.counter = 0
        self.vectorized = vectorized

    def next_context(self):
        """
        Increments the counter.
        """
        self.counter += 1

    def _prepare_site(self, msg):
        """
        Construct the message that is consumed by ``TracePoutine``;
        ``cond_indep_stack`` encodes the nested sequence of ``irange`` branches
        that the site at name is within.
        """
        msg["cond_indep_stack"].append(CondIndepStackFrame(self.name, self.counter, self.vectorized))
        return None

    def _process_message(self, msg):
        return None

    def _postprocess_message(self, msg):
        msg["cond_indep_stack"] = msg["cond_indep_stack"][:-1]
        return None
