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

    def _enter_poutine(self, *args, **kwargs):
        """
        increment counter by one each time we enter a new map_data branch
        """
        self.counter += 1

    def _annotate_map_data_stack(self, msg, name):
        """
        construct the message that is consumed by TraceGraphPoutine
        """
        if len(msg['map_data_stack']) == 0 or msg['map_data_stack'][0] != self.name:
            msg['map_data_stack'].append((self.name, self.counter))
        return msg

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        pack the message with extra information and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        msg = self._annotate_map_data_stack(msg, name)
        ret = super(LambdaPoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)
        return ret

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        pack the message with extra information and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        msg = self._annotate_map_data_stack(msg, name)
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
