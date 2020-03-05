# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

import funsor

import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro.poutine.messenger import Messenger

from pyro.contrib.funsor import to_data, to_funsor, markov
from pyro.contrib.funsor.enum_messenger import EnumMessenger

funsor.set_backend("torch")


class FunsorLogJointMessenger(Messenger):

    def __enter__(self):
        self.log_joint = to_funsor(0., funsor.reals())
        return super().__enter__()

    def _pyro_post_sample(self, msg):
        with funsor.interpreter.interpretation(funsor.terms.lazy):
            funsor_dist = to_funsor(msg["fn"], funsor.reals())
            self.log_joint += funsor_dist(value=to_funsor(msg["value"], funsor_dist.inputs["value"]))


def log_z(model):

    def _wrapped(*args, **kwargs):
        with FunsorLogJointMessenger() as tr:
            with EnumMessenger():
                model(*args)

        with funsor.interpreter.interpretation(funsor.terms.normalize):
            expr = tr.log_joint.reduce(funsor.ops.logaddexp)

        return to_data(funsor.optimizer.apply_optimizer(expr))

    return _wrapped


@config_enumerate
def model(data):

    p = pyro.param("probs", lambda: torch.rand((3, 3)), constraint=constraints.simplex)
    locs = pyro.param("locs", lambda: torch.randn(3))

    x = 0
    for i in markov(range(len(data))):
        x = pyro.sample(f"x{i}", dist.Categorical(p[x]))
        pyro.sample(f"y{i}", dist.Normal(locs[x], 1.), obs=data[i])


def main():
    data = [torch.tensor(1.)] * 10
    log_marginal = log_z(model)(data)
    print(log_marginal)


if __name__ == "__main__":
    main()
