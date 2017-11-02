from __future__ import absolute_import, division, print_function

import pytest

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.util import ng_ones, ng_zeros, set_rng_seed
from tests.common import assert_equal

EXAMPLE_MODELS = []
EXAMPLE_MODEL_IDS = []


class ExampleModel(object):
    def __init__(self, fn, poutine_kwargs):
        self.fn = fn
        self.poutine_kwargs = poutine_kwargs

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def bind_poutine(self, poutine_name):
        """
        Bind model-specific kwargs to the poutine.
        """
        p = getattr(poutine, poutine_name)
        kwargs = self.poutine_kwargs.get(poutine_name, {})
        return lambda fn: p(fn, **kwargs)


def register_model(**poutine_kwargs):
    """
    Decorator to register a model as an example model for testing.
    """

    def register_fn(fn):
        model = ExampleModel(fn, poutine_kwargs)
        EXAMPLE_MODELS.append(model)
        EXAMPLE_MODEL_IDS.append(model.fn.__name__)
        return model

    return register_fn


@register_model(replay={'trace': poutine.Trace()},
                block={},
                condition={'data': {}},
                do={'data': {}})
def trivial_model():
    return []


tr_normal = poutine.Trace()
tr_normal.add_node("normal_0", type="sample", is_observed=False, value=ng_zeros(1))


@register_model(replay={'trace': tr_normal},
                block={'hide': ['normal_0']},
                condition={'data': {'normal_0': ng_zeros(1)}},
                do={'data': {'normal_0': ng_zeros(1)}})
def normal_model():
    normal_0 = pyro.sample('normal_0', dist.normal, ng_zeros(1), ng_ones(1))
    return [normal_0]


tr_normal_normal = poutine.Trace()
tr_normal_normal.add_node("normal_0", type="sample", is_observed=False, value=ng_zeros(1))


@register_model(replay={'trace': tr_normal_normal},
                block={'hide': ['normal_0']},
                condition={'data': {'normal_0': ng_zeros(1)}},
                do={'data': {'normal_0': ng_zeros(1)}})
def normal_normal_model():
    normal_0 = pyro.sample('normal_0', dist.normal, ng_zeros(1), ng_ones(1))
    normal_1 = ng_ones(1)
    pyro.observe('normal_1', dist.normal, normal_1, normal_0, ng_ones(1))
    return [normal_0, normal_1]


tr_bernoulli_normal = poutine.Trace()
tr_bernoulli_normal.add_node("bern_0", type="sample", is_observed=False, value=ng_ones(1))


@register_model(replay={'trace': tr_bernoulli_normal},
                block={'hide': ['bern_0']},
                condition={'data': {'bern_0': ng_ones(1)}},
                do={'data': {'bern_0': ng_ones(1)}})
def bernoulli_normal_model():
    bern_0 = pyro.sample('bern_0', dist.bernoulli, ng_zeros(1) * 1e-2)
    mu = ng_ones(1) if bern_0.data[0] else -ng_ones(1)
    normal_0 = ng_ones(1)
    pyro.observe('normal_0', dist.normal, normal_0, mu, ng_ones(1) * 1e-2)
    return [bern_0, normal_0]


def get_trace(fn, *args, **kwargs):
    set_rng_seed(123)
    return poutine.trace(fn).get_trace(*args, **kwargs)


@pytest.mark.parametrize('model', EXAMPLE_MODELS, ids=EXAMPLE_MODEL_IDS)
@pytest.mark.parametrize('poutine_name', [
    'block',
    'do',
    'replay',
    'trace',
])
def test_idempotent(poutine_name, model):
    p = model.bind_poutine(poutine_name)
    expected_trace = get_trace(p(model))
    actual_trace = get_trace(p(p(model)))
    assert_equal(actual_trace, expected_trace, prec=0)


@pytest.mark.parametrize('model', EXAMPLE_MODELS, ids=EXAMPLE_MODEL_IDS)
@pytest.mark.parametrize('p1_name,p2_name', [
    ('trace', 'condition'),
    ('trace', 'do'),
    ('trace', 'replay'),
])
def test_commutes(p1_name, p2_name, model):
    p1 = model.bind_poutine(p1_name)
    p2 = model.bind_poutine(p2_name)
    expected_trace = get_trace(p1(p2(model)))
    actual_trace = get_trace(p2(p1(model)))
    assert_equal(actual_trace, expected_trace, prec=0)
