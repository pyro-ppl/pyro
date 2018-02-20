from __future__ import absolute_import, division, print_function

from .poutine import Messenger, Poutine


class InferConfigMessenger(Messenger):
    """
    Modifies contents of the infer kwarg at sample sites
    """
    def __init__(self, config_fn=None):
        """
        :param config_fn: a callable taking a site and returning an infer dict

        Constructor. Doesn't do much, just stores the stochastic function
        and the config_fn.
        """
        super(InferConfigMessenger, self).__init__()
        self.config_fn = config_fn

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        If self.config_fn is not None, calls self.config_fn on msg
        and stores the result in msg["infer"].

        Otherwise, implements default sampling behavior
        with no additional effects.
        """
        if self.config_fn is not None:
            msg["infer"] = msg.get("infer", {})
            msg["infer"].update(self.config_fn(msg))
        return None

    def _pyro_param(self, msg):
        return None


class InferConfigPoutine(Poutine):
    """
    Modifies contents of the infer kwarg at sample sites
    """
    def __init__(self, fn, config_fn=None):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)
        :param config_fn: a callable taking a site and returning an infer dict

        Constructor. Doesn't do much, just stores the stochastic function
        and the config_fn.
        """
        super(InferConfigPoutine, self).__init__(
            InferConfigMessenger(config_fn=config_fn), fn)
