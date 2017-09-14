import pyro
from .poutine import Poutine


class LambdaPoutine(Poutine):
    def __init__(self, fn, name):
        super(LambdaPoutine, self).__init__(fn)
        self.name = name

    def report(self, s):
        if False:
            print s

    def _enter_poutine(self, *args, **kwargs):
        self.report("inside lambdapoutine enter poutine")
        self.prev_node = self.name + '_split_node'
        self.join_node = self.name + '_join_node'
        super(LambdaPoutine, self)._enter_poutine(*args, **kwargs)

    def _exit_poutine(self, ret_val, *args, **kwargs):
       	self.report("exit lambda poutine")
        pass

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        self.report("inside lambdapoutine sample")
        msg['current_map_data'] = self.name
        msg['__map_data_nodes'] = {self.name: (name, self.prev_node, self.join_node)}
        self.prev_node = name
        return super(LambdaPoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        self.report("inside lambdapoutine observe")
        msg['current_map_data'] = self.name
        msg['__map_data_nodes'] = {self.name: (name, self.prev_node, self.join_node)}
        self.prev_node = name
        return super(LambdaPoutine, self)._pyro_observe(msg, name, fn, obs, *args, **kwargs)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None):
        self.report("inside lambdapoutine map_data")
        return super(LambdaPoutine, self)._pyro_map_data(msg, name, data, fn,
                                                        batch_size=batch_size)
