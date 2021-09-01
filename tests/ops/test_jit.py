# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro.ops.jit
from tests.common import assert_equal


def test_varying_len_args():
    def fn(*args):
        return sum(args)

    jit_fn = pyro.ops.jit.trace(fn)
    examples = [
        [torch.tensor(1.0)],
        [torch.tensor(2.0), torch.tensor(3.0)],
        [torch.tensor(4.0), torch.tensor(5.0), torch.tensor(6.0)],
    ]
    for args in examples:
        assert_equal(jit_fn(*args), fn(*args))


def test_varying_kwargs():
    def fn(x, scale=1.0):
        return x * scale

    jit_fn = pyro.ops.jit.trace(fn)
    x = torch.tensor(1.0)
    for scale in [-1.0, 0.0, 1.0, 10.0]:
        assert_equal(jit_fn(x, scale=scale), fn(x, scale=scale))


def test_varying_unhashable_kwargs():
    def fn(x, config={}):
        return x * config.get(scale, 1.0)

    jit_fn = pyro.ops.jit.trace(fn)
    x = torch.tensor(1.0)
    for scale in [-1.0, 0.0, 1.0, 10.0]:
        config = {"scale": scale}
        assert_equal(jit_fn(x, config=config), fn(x, config=config))
