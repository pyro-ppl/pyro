import pyro
from .poutine import Poutine


class LambdaPoutine(Poutine):
    """
    This poutine has two functions:
    (i)  handle score-rescaling
    (ii) keep track of dependency structure inside of map_data for
         the benefit of TraceGraphPoutine
    """
    def __init__(self, fn, name, scale):
        """
        Constructor: basically default, but store an extra scalar self.scale
        """
        self.name = name
        self.scale = scale
        super(LambdaPoutine, self).__init__(fn)

    def _enter_poutine(self, *args, **kwargs):
        self.join_node = self.name + '__JOIN_NODE'
        self.split_node = self.name + '__SPLIT_NODE'
        self.prev_node = self.split_node

    # construct the message that is consumed by TraceGraphPoutine
    def _enrich_msg(self, msg, name):
        if len(msg['map_data_stack']) == 0 or msg['map_data_stack'][0] != self.name:
            msg['map_data_stack'].append(self.name)
        if len(msg['map_data_stack']) == 1:
            nodes = {'current': name, 'previous': self.prev_node,
                     'join': self.join_node, 'split': self.split_node}
            msg['map_data_nodes'] = nodes
            self.prev_node = name
        return msg

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        pack the message with extra information and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        msg = self._enrich_msg(msg, name)
        ret = super(LambdaPoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)
        return ret

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        pack the message with extra information and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        msg = self._enrich_msg(msg, name)
        ret = super(LambdaPoutine, self)._pyro_observe(msg, name, fn, obs, *args, **kwargs)
        return ret

    def _pyro_param(self, msg, name, *args, **kwargs):
        """
        Scaled param: Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(LambdaPoutine, self)._pyro_param(msg, name, *args, **kwargs)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None, batch_dim=0):
        """
        scaled map_data: Rescale the message and continue
        """
        mapdata_scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        return super(LambdaPoutine, self)._pyro_map_data(msg, name, data,
                                                         LambdaPoutine(fn, name, mapdata_scale),
                                                         batch_size=batch_size,
                                                         batch_dim=batch_dim)
