# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.poutine.runtime


logger = logging.getLogger(__name__)


def test_nested_reset():

    def nested_model():
        pyro.sample("internal0", dist.Bernoulli(0.5))
        with poutine.escape(escape_fn=lambda msg: msg["name"] == "internal2"):
            pyro.sample("internal1", dist.Bernoulli(0.5))
            pyro.sample("internal2", dist.Bernoulli(0.5))
            pyro.sample("internal3", dist.Bernoulli(0.5))

    with poutine.trace() as t2:
        with poutine.block(hide=["internal2"]):
            with poutine.trace() as t1:
                try:
                    nested_model()
                except poutine.NonlocalExit as site_container:
                    site_container.reset_stack()
                    logger.debug(pyro.poutine.runtime._PYRO_STACK)
                    assert "internal1" not in t1.trace
                    assert "internal1" in t2.trace
