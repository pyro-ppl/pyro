from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict

import torch

import pyro.distributions as dist

PYRO_STACK = []
PARAM_STORE = {}


class Messenger(object):
    def __init__(self, fn):
        self.fn = fn

    def __enter__(self):
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


class trace(Messenger):
    def __enter__(self):
        self.trace = OrderedDict()
        return super(trace, self).__enter__()

    def process_msg(self, msg):
        self.trace[msg["name"]] = msg

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process_msg(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


def sample(name, fn, obs=None):
    if not PYRO_STACK:
        return fn()
    msg = dict(type="sample", name=name, fn=fn, value=obs)
    for handler in reversed(PYRO_STACK):
        handler.process_msg(msg)
    if msg["value"] is None:
        msg["value"] = fn()
    return msg["value"]


def param(name, init_value=None):
    value = PARAM_STORE.setdefault(name, init_value)
    value.requires_grad_()
    if not PYRO_STACK:
        return value
    msg = dict(type="param", name=name, value=value)
    for handler in reversed(PYRO_STACK):
        handler.process_msg(msg)
    return msg["value"]


class Adam(object):
    def __init__(self, optim_args):
        self.optim_args = optim_args
        self.optim_objs = {}  # tensor -> optimizer

    def __call__(self, params):
        for param in params:
            if param in self.optim_objs:
                optim = self.optim_objs[param]
            else:
                optim = torch.optim.Adam([param], **self.optim_args)
                self.optim_objs[param] = optim
            optim.step()


class SVI(object):
    def __init__(self, model, guide, optim, loss):
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss

    def step(self, *args, **kwargs):
        loss = self.loss(self.model, self.guide, *args, **kwargs)
        loss.backward()
        params = list(PARAM_STORE.values())
        self.optim(params)
        for p in params:
            p.grad = p.new_zeros(p.shape)
        return loss.item()


def elbo(model, guide, *args, **kwargs):
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    elbo = 0.
    for site in model_trace.values():
        if site["type"] == "sample":
            elbo = elbo + site["fn"].log_prob(site["value"]).sum()
    for site in guide_trace.values():
        if site["type"] == "sample":
            elbo = elbo - site["fn"].log_prob(site["value"]).sum()
    return -elbo


def main(args):

    def model(data):
        loc = sample("loc", dist.Normal(0., 1.))
        sample("obs", dist.Normal(loc, 1.), obs=data)

    def guide(data):
        loc_loc = param("loc_loc", torch.tensor(0.))
        loc_scale = param("loc_scale_log", torch.tensor(0.)).exp()
        sample("loc", dist.Normal(loc_loc, loc_scale))

    data = torch.randn(100) + 3.0

    svi = SVI(model, guide, Adam({'lr': args.learning_rate}), elbo)
    for step in range(args.num_steps):
        loss = svi.step(data)
        if step % 100 == 0:
            print('step {} loss = {}'.format(step, loss))

    for name, value in PARAM_STORE.items():
        print('{} = {}'.format(name, value.detach().cpu().numpy()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    args = parser.parse_args()
    main(args)
