import pytest
import torch
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.enum import scale_trace
from tests.common import assert_equal


def model():
    p = pyro.param("p", Variable(torch.Tensor([0.1, 0.9])))
    mu = pyro.param("mu", Variable(torch.Tensor([-1.0, 1.0])))
    sigma = pyro.param("sigma", Variable(torch.Tensor([2.0, 3.0])))
    x = pyro.sample("x", dist.Bernoulli(p))
    y = pyro.sample("y", dist.DiagNormal(mu, sigma))
    z = pyro.sample("z", dist.DiagNormal(y, sigma))
    return dict(x=x, y=y, z=z)


@pytest.mark.parametrize("trace_type", [
    "trace",
    pytest.mark.xfail("tracegraph", reason="TraceGraph.copy() is not implemented"),
])
def test_scale_trace(trace_type):
    pyro.get_param_store().clear()
    scale = 1.234
    poutine_trace = getattr(poutine, trace_type)
    tr1 = poutine_trace(model).get_trace()
    tr2 = scale_trace(tr1, scale)

    assert type(tr1) is type(tr2)
    assert tr1 is not tr2, "scale_trace() did not make a copy"
    assert tr1.keys() == tr2.keys()
    for name, site1 in tr1.items():
        if "scale" in site1:
            site2 = tr2[name]
            assert_equal(site2["scale"], scale * site1["scale"], msg=(site1, site2))

    # These check that memoized values were cleared.
    assert_equal(tr2.log_pdf(), scale * tr1.log_pdf())
    assert_equal(tr2.batch_log_pdf(), scale * tr1.batch_log_pdf())
