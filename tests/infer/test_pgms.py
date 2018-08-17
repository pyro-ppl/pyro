import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDelta, AutoDiscreteParallel
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
    def model(data):
        cpd_a = pyro.param("cpd_a", torch.ones(2), constraint=constraints.simplex)
        cpd_b = pyro.param("cpd_b", torch.ones(2, 2) / 2., constraint=constraints.simplex)
        cpd_c = pyro.param("cpd_c", torch.ones(2, 2, 2) / 2., constraint=constraints.simplex)
        cpd_d = pyro.param("cpd_d", torch.ones(2, 2, 2) / 2., constraint=constraints.simplex)
        cpd_e = pyro.param("cpd_e", torch.ones(2, 2) / 2., constraint=constraints.simplex)
        cpd_f = pyro.param("cpd_f", torch.ones(2, 2, 2) / 2., constraint=constraints.simplex)
        a = pyro.sample("a", dist.Categorical(cpd_a))
        b = pyro.sample("b", dist.Categorical(cpd_b[a]))
        with pyro.iarange("data_c", len(data), dim=-1):
            c = pyro.sample("c", dist.Categorical(cpd_c[a, b]), obs=data[:, 0])
            d = pyro.sample("d", dist.Categorical(cpd_d[b, c.long()]))
            e = pyro.sample("e", dist.Categorical(cpd_e[d]))
        with pyro.iarange("data_f", len(data), dim=-2):
            pyro.sample("f", dist.Categorical(cpd_f[d, e]), obs=data[:, 1])

    @poutine.broadcast
    @config_enumerate(default="parallel", expand=False)
    def guide(data):
        cpd_a = pyro.param("cpd_a", torch.ones(2), constraint=constraints.simplex)
        cpd_b = pyro.param("cpd_b", torch.ones(2, 2) / 2., constraint=constraints.simplex)
        cpd_d = pyro.param("cpd_d", torch.ones(2, 2, 2) / 2., constraint=constraints.simplex)
        cpd_e = pyro.param("cpd_e", torch.ones(2, 2) / 2., constraint=constraints.simplex)
        a = pyro.sample("a", dist.Categorical(cpd_a))
        b = pyro.sample("b", dist.Categorical(cpd_b[a]))
        with pyro.iarange("data_c", len(data), dim=-1):
            d = pyro.sample("d", dist.Categorical(cpd_d[b, data[:, 0].long()]))
            e = pyro.sample("e", dist.Categorical(cpd_e[d]))

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
    adam = optim.Adam({"lr": .01, "betas": (0.95, 0.999)})
    svi = SVI(model, guide, adam, loss=TraceEnum_ELBO(max_iarange_nesting=2))
    data = torch.ones(1000, 2)

    for i in range(3000):
        if i % 100:
            print(svi.step(data))
    print(pyro.param("cpd_a"))
    print(pyro.param("cpd_b"))
    print(pyro.param("cpd_c"))
    print(pyro.param("cpd_d"))
    print(pyro.param("cpd_e"))
    print(pyro.param("cpd_f"))
