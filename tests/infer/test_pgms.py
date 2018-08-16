import torch

import pyro
import pyro.distributions as dist
from pyro.infer import TraceEnum_ELBO, SVI, config_enumerate
import pyro.optim as optim
import pyro.poutine as poutine


def test_discrete_dag():
    """
    a --. b
    |  /  |
    ..    .
    c -- .d --. e
          |  /
          . .
          f
    """

    @poutine.broadcast
    @config_enumerate(default="parallel", expand=False)
    def model(data_c=None, data_f=None):
        cpd_b = pyro.param("cpd_b", torch.ones(2) / 2.)
        cpd_c = pyro.param("cpd_c", torch.ones(2, 2) / 2.)
        cpd_d = pyro.param("cpd_d", torch.ones(2, 2) / 2.)
        cpd_e = pyro.param("cpd_e", torch.ones(2) / 2.)
        cpd_f = pyro.param("cpd_f", torch.ones(2, 2) / 2.)
        a = pyro.sample("a", dist.Bernoulli(0.5))
        b = pyro.sample("b", dist.Bernoulli(cpd_b[a.long()]))
        size_data = len(data_c) if data_c is not None else 1
        with pyro.iarange("data_c", size_data):
            c = pyro.sample("c", dist.Bernoulli(cpd_c[a.long()][b.long()]), obs=data_c)
        d = pyro.sample("d", dist.Bernoulli(cpd_d[b.long()][c.long()]))
        e = pyro.sample("e", dist.Bernoulli(cpd_e[d.long()]))
        with pyro.iarange("data_f", size_data):
            pyro.sample("f", dist.Bernoulli(cpd_f[d.long()][e.long()]), obs=data_f)

    def conditioned_model(data):
        return poutine.condition(model, data=data)()

    def generate_data(n=1000):
        pyro.param("cpd_b", torch.tensor([0.3, 0.7]))
        pyro.param("cpd_c", torch.tensor([[0.2, 0.8], [0.4, 0.6]]))
        pyro.param("cpd_d", torch.tensor([[0.5, 0.5], [0.1, 0.9]]))
        pyro.param("cpd_e", torch.tensor([0.37, 0.63]))
        pyro.param("cpd_f", torch.tensor([[0.25, 0.75], [0.8, 0.2]]))
        c, f = [], []
        for _ in range(1000):
            trace = poutine.trace(model).get_trace()
            c.append(trace.nodes["c"]["value"])
            f.append(trace.nodes["f"]["value"])
        return torch.stack(c), torch.stack(f)

    pyro.clear_param_store()
    adam = optim.Adam({"lr": .0005, "betas": (0.95, 0.999)})
    svi = SVI(model, model, adam, loss=TraceEnum_ELBO(max_iarange_nesting=0))
    data = dict(zip(("c", "f"), generate_data()))

    for _ in range(1000):
        print(svi.step(data))
    print(pyro.param("cpd_b"))