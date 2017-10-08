import pytest

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.util import ng_zeros, ng_ones
from tests.common import assert_equal, freeze_rng_state

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


@register_model(replay={'trace': {}},
                block={},
                condition={'data': {}},
                do={'data': {}})
def trivial_model():
    return []


@register_model(replay={'trace': {'normal_0': {'type': 'sample',
                                               'value': ng_zeros(1)}}},
                block={'hide': ['normal_0']},
                condition={'data': {'normal_0': ng_zeros(1)}},
                do={'data': {'normal_0': ng_zeros(1)}})
def diagnormal_model():
    diagnormal_0 = pyro.sample('normal_0', dist.diagnormal, ng_zeros(1), ng_ones(1))
    return [diagnormal_0]


def get_trace(fn, *args, **kwargs):
    with freeze_rng_state():
        return poutine.trace(fn).get_trace(*args, **kwargs)


@pytest.mark.parametrize('model', EXAMPLE_MODELS, ids=EXAMPLE_MODEL_IDS)
@pytest.mark.parametrize('poutine_name', [
    'trace',
    'tracegraph',
    'replay',
    'block',
    'do',
])
def test_idempotent(poutine_name, model):
    p = model.bind_poutine(poutine_name)
    expected_trace = get_trace(p(model))
    actual_trace = get_trace(p(p(model)))
    assert_equal(actual_trace, expected_trace, prec=0)
