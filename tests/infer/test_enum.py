import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.enum import iter_discrete_traces, scale_trace
from pyro.poutine.trace import TraceGraph
from tests.common import assert_equal


def model1():
    p = pyro.param("p", Variable(torch.Tensor([0.1, 0.9])))
    mu = pyro.param("mu", Variable(torch.Tensor([-1.0, 1.0])))
    sigma = pyro.param("sigma", Variable(torch.Tensor([2.0, 3.0])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.DiagNormal(mu, sigma))
    z = pyro.sample("z", dist.DiagNormal(y, sigma))
    return dict(x=x, y=y, z=z)


def model2():
    p = pyro.param("p", Variable(torch.Tensor([0.05])))
    ps = pyro.param("ps", Variable(torch.Tensor([0.1, 0.2, 0.3, 0.4])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.Categorical(ps, one_hot=False))
    return dict(x=x, y=y)


@pytest.mark.parametrize("model", [model1, model2])
@pytest.mark.parametrize("trace_type", ["trace", "tracegraph"])
def test_scale_trace(trace_type, model):
    pyro.get_param_store().clear()
    scale = 1.234
    poutine_trace = getattr(poutine, trace_type)
    tr1 = poutine_trace(model).get_trace()
    tr2 = scale_trace(tr1, scale)

    assert type(tr1) is type(tr2)
    assert tr1 is not tr2, "scale_trace() did not make a copy"
    if isinstance(tr1, TraceGraph):
        tr1 = tr1.trace
        tr2 = tr2.trace

    assert set(tr1.keys()) == set(tr2.keys())
    for name, site1 in tr1.items():
        if "scale" in site1:
            site2 = tr2[name]
            assert_equal(site2["scale"], scale * site1["scale"], msg=(site1, site2))

    # These check that memoized values were cleared.
    assert_equal(tr2.log_pdf(), scale * tr1.log_pdf())
    assert_equal(tr2.batch_log_pdf(), scale * tr1.batch_log_pdf())


@pytest.mark.parametrize("trace_type", ["trace", "tracegraph"])
def test_iter_discrete_traces(trace_type):
    pyro.get_param_store().clear()
    poutine_trace = getattr(poutine, trace_type)
    traces = list(iter_discrete_traces(poutine_trace, model2))

    p = pyro.param("p").data
    ps = pyro.param("ps").data
    assert len(traces) == 2 * len(ps)

    for scale, trace in traces:
        if isinstance(trace, TraceGraph):
            trace = trace.trace
        x = trace["x"]["value"].data.squeeze().long()[0]
        y = trace["y"]["value"].data.squeeze().long()[0]
        expected_scale = [1 - p[0], p[0]][x] * ps[y]
        assert_equal(scale, expected_scale)
