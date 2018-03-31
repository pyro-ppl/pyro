from __future__ import absolute_import, division, print_function

from .poutine import Messenger, Poutine
from .indep_poutine import IndepMessenger


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
        self._ctxs = []

    def __enter__(self):
        """
        TODO docs
        """
        self.next_available_dim = self.first_available_dim
        return super(ContinuationMessenger, self).__enter__()

    def __exit__(self, *args):
        """
        TODO docs
        """
        for ctx in reversed(self._ctxs):
            ctx.__exit__(*args)
        return super(ContinuationMessenger, self).__exit__(*args)

    def _reset(self):
        """
        TODO docs
        """
        self._ctxs = []

    def _postprocess_message(self, msg):
        """
        TODO docs
        """
        if "next_available_dim" in msg["infer"]:
            self.next_available_dim = msg["infer"]["next_available_dim"]
        for ctx in self._ctxs:
            if msg["type"] == "sample" and \
               msg["fn"].batch_shape[ctx.frame.dim] == 1:
                ctx.__exit__()
                msg["cond_indep_stack"].pop(ctx.frame)

    def _process_message(self, msg):
        """
        TODO docs
        """
        if self.escape_fn(msg) and not msg["done"]:
            msg["infer"]["next_available_dim"] = self.next_available_dim
            msg["done"] = True
            msg["continuation"] = self.cont_fn

            ctx = IndepMessenger("ctx_{}".format(msg["name"]),
                                 size=1,  # external=True,
                                 dim=self.next_available_dim)
            ctx.__enter__()
            ctx._process_message(msg)
            self._ctxs.append(ctx)


class ContinuationPoutine(Poutine):
    def __init__(self, fn, escape_fn, cont_fn, first_available_dim=None):
        super(ContinuationPoutine, self).__init__(
            ContinuationMessenger(escape_fn, cont_fn, first_available_dim), fn)
