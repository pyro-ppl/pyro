import pyro
from pyro import poutine
from pyro.nn import PyroModule


class Model(PyroModule):
    num_treatments = 2

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            z = pyro.sample("z", self.z_dist)
            x = pyro.sample("x", self.x_dist(z), obs=x)
            t = pyro.sample("t", self.t_dist(z), obs=t)
            y = pyro.sample("y", self.y_dist(t, z), obs=y)
        return y

    def z_dist(self):
        raise NotImplementedError

    def x_dist(self, z):
        raise NotImplementedError

    def y_dist(self, t, y):
        raise NotImplementedError


class Guide(PyroModule):
    num_treatments = 2

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            t = pyro.sample("t", self.t_dist(x), obs=t)
            y = pyro.sample("y", self.y_dist(t, x), obs=y)
            pyro.sample("z", self.z_dist(t, x, y))

    def t_dist(self, x):
        raise NotImplementedError

    def y_dist(self, t, y):
        raise NotImplementedError

    def z_dist(self, t, y, x):
        raise NotImplementedError


def ite(model, guide, x, num_samples=100):
    """
    Individual treatment effect.
    """
    with pyro.plate("particles", num_samples, dim=-2):
        with poutine.do(data=dict(t=0)):
            tr = poutine.trace(guide).get_trace(x)
            y0 = poutine.replay(model, tr)(x)
        with poutine.do(data=dict(t=1)):
            tr = poutine.trace(guide).get_trace(x)
            y1 = poutine.replay(model, tr)(x)
    return (y1 - y0).mean(0)


def main(args):
    raise NotImplementedError
