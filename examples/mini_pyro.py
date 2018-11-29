from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict

import torch

import pyro.distributions as dist

PYRO_STACK = []
PARAM_STORE = {}


class Messenger(object):
    def __init__(self, fn=None):
        self.fn = fn

    def __enter__(self):
        PYRO_STACK.append(self)

    def __exit__(self, *args, **kwargs):
        assert PYRO_STACK[-1] is self
        PYRO_STACK.pop()

    def process_message(self, msg):
        pass

    def postprocess_message(self, msg):
        pass

    def __call__(self, *args, **kwargs):
        with self:
            return self.fn(*args, **kwargs)


class trace(Messenger):
    def __enter__(self):
        super(trace, self).__enter__()
        self.trace = OrderedDict()
        return self.trace

    def postprocess_message(self, msg):
        self.trace[msg["name"]] = msg.copy()

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.trace


class replay(Messenger):
    def __init__(self, fn, guide_trace):
        self.guide_trace = guide_trace
        super(replay, self).__init__(fn)

    def process_message(self, msg):
        if msg["name"] in self.guide_trace:
            msg["value"] = self.guide_trace[msg["name"]]["value"]


def sample(name, fn, obs=None):
    if not PYRO_STACK:
        return fn()
    msg = dict(type="sample", name=name, fn=fn, value=obs)
    for handler in reversed(PYRO_STACK):
        handler.process_message(msg)
    if msg["value"] is None:
        msg["value"] = fn()
    for handler in PYRO_STACK:
        handler.postprocess_message(msg)
    return msg["value"]


def param(name, init_value=None):
    value = PARAM_STORE.setdefault(name, init_value)
    value.requires_grad_()
    if not PYRO_STACK:
        return value
    msg = dict(type="param", name=name, value=None)
    for handler in reversed(PYRO_STACK):
        handler.process_message(msg)
    if msg["value"] is None:
        msg["value"] = value
    for handler in PYRO_STACK:
        handler.postprocess_message(msg)
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
        with trace() as param_capture:
            loss = self.loss(self.model, self.guide, *args, **kwargs)
        loss.backward()
        params = [site["value"]
                  for site in param_capture.values()
                  if site["type"] == "param"]
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
    torch.manual_seed(0)

    def model(data):
        loc = sample("loc", dist.Normal(0., 1.))
        sample("obs", dist.Normal(loc, 1.), obs=data)

    def guide(data):
        loc_loc = param("loc_loc", torch.tensor(0.))
        loc_scale = param("loc_scale_log", torch.tensor(0.)).exp()
        sample("loc", dist.Normal(loc_loc, loc_scale))

    data = torch.randn(100) + 3.0

    svi = SVI(model, guide, Adam({"lr": args.learning_rate}), elbo)
    for step in range(args.num_steps):
        loss = svi.step(data)
        if step % 100 == 0:
            print("step {} loss = {}".format(step, loss))

    for name, value in PARAM_STORE.items():
        print("{} = {}".format(name, value.detach().cpu().numpy()))

    assert (param("loc_loc") - 3.0).abs() < 0.1


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.0')
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    args = parser.parse_args()
    main(args)
