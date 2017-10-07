from .util import NonlocalExit
from .poutine import Poutine


class EscapePoutine(Poutine):
    """
    TODO doc
    """
    def __init__(self, fn, escape_fn):
        """
        TODO doc
        """
        self.escape_fn = escape_fn
        super(EscapePoutine, self).__init__(fn)

    def _pyro_sample(self, msg):
        """
        TODO doc
        """
        if self.escape_fn(msg):
            msg["done"] = True
            raise NonlocalExit(msg)
        else:
            return super(EscapePoutine, self)._pyro_sample(msg)
