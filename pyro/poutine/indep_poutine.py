from __future__ import absolute_import, division, print_function

from collections import namedtuple

from .poutine import Poutine

CondIndepStackFrame = namedtuple("CondIndepStackFrame", ["name", "counter", "vectorized"])


class IndepPoutine(Poutine):
    """
    This poutine keeps track of stack of independence information declared by
    nested ``irange`` and ``iarange`` contexts. This information is stored in
    a ``cond_indep_stack`` at each sample/observe site for consumption by
    ``TracePoutine``.
    """
    def __init__(self, fn, name, vectorized):
        """
        Constructor: basically default, but store a counter to keep track of
        which ``irange`` branch we're in.
        """
        self.name = name
        self.counter = 0
        self.vectorized = vectorized
        super(IndepPoutine, self).__init__(fn)

    def __enter__(self):
        """
        Increment counter by one each time we enter a new ``irange`` branch.
        """
        self.counter += 1
        return super(IndepPoutine, self).__enter__()

    def _prepare_site(self, msg):
        """
        Construct the message that is consumed by ``TracePoutine``;
        ``cond_indep_stack`` encodes the nested sequence of ``irange`` branches
        that the site at name is within.
        """
        msg["cond_indep_stack"].append(CondIndepStackFrame(self.name, self.counter, self.vectorized))
        return msg
