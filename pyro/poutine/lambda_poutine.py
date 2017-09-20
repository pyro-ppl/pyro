import pyro
from .poutine import Poutine


class LambdaPoutine(Poutine):
    """
    score-rescaling Poutine
    Subsampling means we have to rescale pdfs inside map_data
    This poutine handles the rescaling because it wouldn't fit in Poutine
    """
    def __init__(self, fn, name, scale):
        """
        Constructor: basically default, but store an extra scalar self.scale
        """
        self.name = name
        self.scale = scale
        super(LambdaPoutine, self).__init__(fn)

    def _enter_poutine(self, *args, **kwargs):
        self.report("[%s] Enter LambdaPoutine" % self.name)
        self.join_node = self.name + '_join_node'
        self.split_node = self.name + '_split_node'
        self.prev_node = self.split_node
        #super(LambdaPoutine, self)._enter_poutine(*args, **kwargs)

    def report(self, s):
        if True:
            print s

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Scaled sampling: Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        if len(msg['map_data_stack']) == 0 or msg['map_data_stack'][0] != self.name:
            msg['map_data_stack'].append(self.name)
        nodes = {'current': name, 'previous': self.prev_node,
                'join': self.join_node, 'split': self.split_node}
        #msg['map_data_nodes'][self.name] = nodes
        if len(msg['map_data_stack']) == 1:
            msg['map_data_nodes'][self.name] = nodes
            self.prev_node = name
        self.report("[%s/%s] Exit LambdaPoutine OBSERVE" % (self.name, name))
        ret = super(LambdaPoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)
        return ret

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        Scaled observe: Rescale the message and continue
        """
        self.report("[%s/%s] Enter LambdaPoutine OBSERVE" % (self.name, name))
        if len(msg['map_data_stack']) == 0 or msg['map_data_stack'][0] != self.name:
            msg['map_data_stack'].append(self.name)
        nodes = {'current': name, 'previous': self.prev_node,
                'join': self.join_node, 'split': self.split_node}
        #msg['map_data_nodes'][self.name] = nodes
        if len(msg['map_data_stack']) == 1:
            msg['map_data_nodes'][self.name] = nodes
            self.prev_node = name
        msg["scale"] = self.scale * msg["scale"]
        ret = super(LambdaPoutine, self)._pyro_observe(msg, name, fn, obs, *args, **kwargs)
        self.report("[%s/%s] Exit LambdaPoutine OBSERVE" % (self.name, name))
	return ret

    def _pyro_param(self, msg, name, *args, **kwargs):
        """
        Scaled param: Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(LambdaPoutine, self)._pyro_param(msg, name, *args, **kwargs)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None, batch_dim=0):
        """
        Scaled map_data: Rescale the message and continue
        Should just work...
        """
        mapdata_scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        return super(LambdaPoutine, self)._pyro_map_data(msg, name, data,
                                                        LambdaPoutine(fn, name, mapdata_scale),
                                                        batch_size=batch_size,
                                                        batch_dim=batch_dim)
