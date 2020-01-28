# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from .messenger import Messenger


class InferConfigMessenger(Messenger):
    """
    Given a callable `fn` that contains Pyro primitive calls
    and a callable `config_fn` taking a trace site and returning a dictionary,
    updates the value of the infer kwarg at a sample site to config_fn(site).

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param config_fn: a callable taking a site and returning an infer dict
    :returns: stochastic function decorated with :class:`~pyro.poutine.infer_config_messenger.InferConfigMessenger`
    """
    def __init__(self, config_fn):
        """
        :param config_fn: a callable taking a site and returning an infer dict

        Constructor. Doesn't do much, just stores the stochastic function
        and the config_fn.
        """
        super().__init__()
        self.config_fn = config_fn

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        If self.config_fn is not None, calls self.config_fn on msg
        and stores the result in msg["infer"].

        Otherwise, implements default sampling behavior
        with no additional effects.
        """
        msg["infer"].update(self.config_fn(msg))
        return None

    def _pyro_param(self, msg):
        """
        :param msg: current message at a trace site.

        If self.config_fn is not None, calls self.config_fn on msg
        and stores the result in msg["infer"].

        Otherwise, implements default param behavior
        with no additional effects.
        """
        msg["infer"].update(self.config_fn(msg))
        return None
