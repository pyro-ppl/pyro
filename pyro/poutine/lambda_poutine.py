from .poutine import Poutine


class LambdaPoutine(Poutine):
    """
    This poutine has two functions:
    (i)  handle score-rescaling
    (ii) keep track of stack of nested map_datas at each sample/observe site
         for the benefit of TraceGraphPoutine;
         necessary information passed via map_data_stack in msg
    """
    def __init__(self, fn, name, scale):
        """
        Constructor: basically default, but store an extra scalar self.scale
        and a counter to keep track of which (list) map_data branch we're in
        """
        self.name = name
        self.scale = scale
        self.counter = 0
        super(LambdaPoutine, self).__init__(fn)

    def _enter_poutine(self, *args, **kwargs):
        """
        increment counter by one each time we enter a new map_data branch
        """
        self.counter += 1

    def down(self, msg):
        """
        construct the message that is consumed by TraceGraphPoutine;
        map_data_stack encodes the nested sequence of map_data branches
        that the site at name is within
        """
        msg["scale"] = self.scale * msg["scale"]
        if len(msg['map_data_stack']) == 0 or msg['map_data_stack'][0] != self.name:
            msg['map_data_stack'].insert(0, (self.name, self.counter))
        return msg
