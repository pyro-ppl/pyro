# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Callable

from pyro.poutine.messenger import Messenger

if TYPE_CHECKING:
    from pyro.poutine.runtime import InferDict, Message


class InferConfigMessenger(Messenger):
    """
    Given a callable `fn` that contains Pyro primitive calls
    and a callable `config_fn` taking a trace site and returning a dictionary,
    updates the value of the infer kwarg at a sample site to config_fn(site).

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param config_fn: a callable taking a site and returning an infer dict
    :returns: stochastic function decorated with :class:`~pyro.poutine.infer_config_messenger.InferConfigMessenger`
    """

    def __init__(self, config_fn: Callable[["Message"], "InferDict"]) -> None:
        """
        :param config_fn: a callable taking a site and returning an infer dict

        Constructor. Doesn't do much, just stores the stochastic function
        and the config_fn.
        """
        super().__init__()
        self.config_fn = config_fn

    def _pyro_sample(self, msg: "Message") -> None:
        """
        :param msg: current message at a trace site.

        If self.config_fn is not None, calls self.config_fn on msg
        and stores the result in msg["infer"].

        Otherwise, implements default sampling behavior
        with no additional effects.
        """
        assert msg["infer"] is not None
        msg["infer"].update(self.config_fn(msg))

    def _pyro_param(self, msg: "Message") -> None:
        """
        :param msg: current message at a trace site.

        If self.config_fn is not None, calls self.config_fn on msg
        and stores the result in msg["infer"].

        Otherwise, implements default param behavior
        with no additional effects.
        """
        assert msg["infer"] is not None
        msg["infer"].update(self.config_fn(msg))
