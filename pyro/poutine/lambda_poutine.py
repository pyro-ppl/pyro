from __future__ import absolute_import, division, print_function

from collections import namedtuple

from .poutine import Poutine

CondIndepStackFrame = namedtuple("CondIndepStackFrame", ["name", "counter", "vectorized"])


class LambdaPoutine(Poutine):
    """
    This poutine has two functions:
        (i)  handle score-rescaling
        (ii) keep track of stack of nested map_datas at each sample/observe site
             for the benefit of TracePoutine;
             necessary information passed via map_data_stack in msg
    """
    def __init__(self, fn, name, scale, vectorized):
        """
        Constructor: basically default, but store an extra scalar self.scale
        and a counter to keep track of which (list) map_data branch we're in
        """
        self.name = name
        self.scale = scale
        self.counter = 0
        self.vectorized = vectorized
        super(LambdaPoutine, self).__init__(fn)

    def __enter__(self):
        """
        increment counter by one each time we enter a new map_data branch
        """
        self.counter += 1
        return super(LambdaPoutine, self).__enter__()

    def _prepare_site(self, msg):
        """
        construct the message that is consumed by TracePoutine;
        map_data_stack encodes the nested sequence of map_data branches
        that the site at name is within
        note: the map_data_stack ordering is innermost to outermost from left to right
        """
        msg["scale"] = self.scale * msg["scale"]
        msg["map_data_stack"].insert(0, CondIndepStackFrame(self.name, self.counter, self.vectorized))
        return msg
