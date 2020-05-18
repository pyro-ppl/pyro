# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide.initialization import InitMessenger, init_to_generated, init_to_value


def test_init_to_generated():
    def model():
        x = pyro.sample("x", dist.Normal(0, 1))
        y = pyro.sample("y", dist.Normal(0, 1))
        z = pyro.sample("z", dist.Normal(0, 1))
        return x, y, z

    class MockGenerate:
        def __init__(self):
            self.counter = 0

        def __call__(self):
            values = {"x": torch.tensor(self.counter + 0.0),
                      "y": torch.tensor(self.counter + 0.5)}
            self.counter += 1
            return init_to_value(values=values)

    mock_generate = MockGenerate()
    with InitMessenger(init_to_generated(generate=mock_generate)):
        for i in range(5):
            x, y, z = model()
            assert x == i
            assert y == i + 0.5
    assert mock_generate.counter == 5
