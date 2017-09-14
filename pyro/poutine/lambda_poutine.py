import pyro
from .poutine import Poutine


class LambdaPoutine(Poutine):
    def __init__(self, fn, name):
        super(LambdaPoutine, self).__init__(fn)
        self.name = name
        #self.counter = 0

    def report(self, s):
        if False:
            print s

    #def __call__(self, *args, **kwargs):
        #print "inside lambdapoutine call; args[0]=%d" % args[0]
        #self.lambda_counter += 1
    #    return super(LambdaPoutine, self).__call__(*args, **kwargs)

    def _enter_poutine(self, *args, **kwargs):
        #self.counter += 1
        self.prev_node = self.name + '_split_node'
        self.join_node = self.name + '_join_node'
        #self.prev_node = self.name + ('_split_node_%d' % self.counter)
        #self.join_node = self.name + ('_join_node_%d' % self.counter)
       	#self.report("enter lambda poutine %d" % self.counter)
        super(LambdaPoutine, self)._enter_poutine(*args, **kwargs)

    def _exit_poutine(self, ret_val, *args, **kwargs):
       	self.report("exit lambda poutine")
        pass

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        self.report("inside lambdapoutine sample")
        return super(LambdaPoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        #print "inside lambdapoutine observe; msg:", msg['type'], msg.keys()
        #print "inside lambdapoutine observe; counter:",  self.counter
        #print "[%s] msg.keys" % name, msg.keys()
        #if '__map_data_current_nodes' not in msg:
        #    msg['__map_data_current_nodes'] = {}
        #msg['__map_data_current_nodes'][self.name] = name
        #if '__map_data_counters' not in msg:
        #    msg['__map_data_counters'] = {}
        #if '__map_data_previous_nodes' not in msg:
        #    msg['__map_data_previous_nodes'] = {self.name: self.name + '_split_node'}
        #else:
        #    msg['__map_data_previous_nodes'][self.name] = msg['__map_data_current_nodes'][self.name]
        msg['current_map_data'] = self.name
        #msg['__map_data_lambda_counters'] = {self.name: self.counter}
        msg['__map_data_nodes'] = {self.name: (name, self.prev_node, self.join_node)}
        self.prev_node = name
        #msg['__map_data_previous_nodes'][self.name] = msg['__map_data_current_nodes'][self.name]
        return super(LambdaPoutine, self)._pyro_observe(msg, name, fn, obs, *args, **kwargs)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None):
        self.report("inside lambdapoutine map_data")
        return super(LambdaPoutine, self)._pyro_map_data(msg, name, data, fn,
                                                        batch_size=batch_size)
