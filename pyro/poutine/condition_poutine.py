import pyro

from .poutine import Poutine


class ConditionPoutine(Poutine):
    """
    Adds values at observe sites to condition on data and override sampling
    """
    def __init__(self, fn, data=None):
        """
        constructor
        data is a dictionary of names and values to condition on
        """
        if data is None:
            data = {}
        self.data = data
        super(ConditionPoutine, self).__init__(fn)

    def _block_down(self, msg):
        if msg["name"] in self.data:
            return True
        else:
            return False

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Here we change the obs argument and thread it up through the stack
        """
        if name in self.data:
            assert kwargs["val"] is None, \
                "ConditionPoutine should not override existing observations"
            msg["type"] = "observe"
            msg["val"] = self.data[name]
            return super(ConditionPoutine, self)._pyro_observe(msg, name, fn,
                                                               self.data[name],
                                                               *args, **kwargs)
        else:
            return super(ConditionPoutine, self)._pyro_sample(msg, name, fn,
                                                              *args, **kwargs)
