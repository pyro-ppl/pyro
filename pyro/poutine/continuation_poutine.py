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
        super(ContinuationMessenger, self).__init__()
        if first_available_dim is None:
            first_available_dim = float('inf')
        self.escape_fn = escape_fn
        self.cont_fn = cont_fn
        self.first_available_dim = first_available_dim
        self.next_available_dim = None
        self._ctxs = []  # OrderedDict({})

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
        for ctx in self._ctxs[:]:  # name, ctx in reversed(self._ctxs.items()):
            ctx.__exit__(*args)
        return super(ContinuationMessenger, self).__exit__(*args)

    def _reset(self):
        """
        TODO docs
        """
        self._ctxs = []  # OrderedDict({})

    def _postprocess_message(self, msg):
        """
        TODO docs
        """

        # if "next_available_dim" in msg["infer"]:
        #     self.next_available_dim = msg["infer"]["next_available_dim"]

        for ctx in reversed(self._ctxs):
            ctx._postprocess_message(msg)

        if msg["type"] == "sample":
            for i, frame in enumerate(msg["cond_indep_stack"]):
                print(frame, msg["value"].shape)
                if frame.name in [ctx.name for ctx in self._ctxs] and \
                   msg["value"].shape[frame.dim] == 1:  # XXX should check msg["fn"]
                    print("here")
                    ctx = list(filter(lambda c: c.name == frame.name, self._ctxs))[0]
                    ctx.__exit__()

    def _process_message(self, msg):
        """
        TODO docs
        """
        if self.escape_fn(msg) and not msg["done"]:
            msg["done"] = True
            msg["continuation"] = self.cont_fn

            # what needs to happen:
            # we are simulating inserting a new iarange just before the site.
            # we need to create an IndepMessenger with appropriate dim,
            # insert it into the stack below all other indepmessengers
            # managed by this messenger,
            # apply it at the first site,
            # exit (pop from stack) just before this messenger exits,
            # and optionally exit early for Markov

            ctx = IndepMessenger("ctx_{}".format(msg["name"]),
                                 size=1,  # external=True,
                                 dim=None,
                                 stack=self._ctxs)  # self.next_available_dim)
            ctx.__enter__()
            for ctx2 in self._ctxs:
                ctx2._process_message(msg)
            # msg["infer"]["next_available_dim"] = ctx.dim  # self.next_available_dim


class ContinuationPoutine(Poutine):
    def __init__(self, fn, escape_fn, cont_fn, first_available_dim=None):
        super(ContinuationPoutine, self).__init__(
            ContinuationMessenger(escape_fn, cont_fn, first_available_dim), fn)
