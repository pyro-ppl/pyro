import pyro
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

    def __enter__(self):
        """
        increment counter by one each time we enter a new map_data branch
        """
        self.counter += 1
        return super(LambdaPoutine, self).__enter__()

    def _annotate_map_data_stack(self, msg, name):
        """
        construct the message that is consumed by TraceGraphPoutine;
        map_data_stack encodes the nested sequence of map_data branches
        that the site at name is within
        """
        if len(msg['map_data_stack']) == 0 or msg['map_data_stack'][0] != self.name:
            msg['map_data_stack'].append((self.name, self.counter))
        return msg

    def _pyro_sample(self, msg):
        """
        pack the message with extra information and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        msg = self._annotate_map_data_stack(msg, msg["name"])
        ret = super(LambdaPoutine, self)._pyro_sample(msg)
        return ret

    def _pyro_observe(self, msg):
        """
        pack the message with extra information and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        msg = self._annotate_map_data_stack(msg, msg["name"])
        ret = super(LambdaPoutine, self)._pyro_observe(msg)
        return ret

    def _pyro_param(self, msg):
        """
        pack the message with extra information and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        msg = self._annotate_map_data_stack(msg, msg["name"])
        return super(LambdaPoutine, self)._pyro_param(msg)

    def _pyro_map_data(self, msg):
        """
        scaled map_data: Rescale the message and continue
        """
        name, data, fn, batch_size, batch_dim = \
            msg["name"], msg["data"], msg["fn"], msg["batch_size"], msg["batch_dim"]
        mapdata_scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        msg.update({"fn": LambdaPoutine(fn, name, mapdata_scale)})
        return super(LambdaPoutine, self)._pyro_map_data(msg)
