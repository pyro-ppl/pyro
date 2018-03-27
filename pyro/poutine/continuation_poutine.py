from __future__ import absolute_import, division, print_function

from .poutine import Messenger, Poutine


class ContinuationMessenger(Messenger):
    """
    TODO docs
    """
    def __init__(self, escape_fn, cont_fn, first_available_dim):
        """
        TODO docs
        """
        if first_available_dim is None:
            first_available_dim = float('inf')
        self.escape_fn = escape_fn
        self.cont_fn = cont_fn
        self.first_available_dim = first_available_dim
        self.next_available_dim = None

    def __enter__(self):
        """
        TODO docs
        """
        self.next_available_dim = self.first_available_dim
        return super(ContinuationMessenger, self).__enter__()

    def _postprocess_message(self, msg):
        if "next_available_dim" in msg["infer"]:
            self.next_available_dim = msg["infer"]["next_available_dim"]

    def _pyro_sample(self, msg):
        """
        TODO docs
        """
        if self.escape_fn(msg) and not msg["done"]:
            msg["infer"]["next_available_dim"] = self.next_available_dim
            msg["done"] = True
            msg["continuation"] = self.cont_fn

    def _pyro_param(self, msg):
        """
        TODO docs
        """
        if self.escape_fn(msg) and not msg["done"]:
            msg["done"] = True
            msg["continuation"] = self.cont_fn


class ContinuationPoutine(Poutine):
    def __init__(self, fn, escape_fn, cont_fn, first_available_dim=None):
        super(ContinuationPoutine, self).__init__(
            ContinuationMessenger(escape_fn, cont_fn, first_available_dim), fn)
