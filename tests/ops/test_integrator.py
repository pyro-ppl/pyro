from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple

import pytest
import torch
from torch.autograd import Variable

from pyro.ops.integrator import velocity_verlet
from tests.common import assert_equal

logger = logging.getLogger(__name__)


TEST_EXAMPLES = []
EXAMPLE_IDS = []

ModelArgs = namedtuple('model_args', ['step_size', 'num_steps', 'q_i', 'p_i', 'q_f', 'p_f', 'prec'])
Example = namedtuple('test_case', ['model', 'args'])


def tensor(arr):
    return Variable(torch.Tensor(arr))


def register_model(init_args):
    """
    Register the model along with each of the model arguments
    as test examples.
    """
    def register_fn(model):
        for args in init_args:
            test_example = Example(model, args)
            TEST_EXAMPLES.append(test_example)
            EXAMPLE_IDS.append(model.__name__)
    return register_fn


@register_model([
    ModelArgs(
        step_size=0.01,
        num_steps=100,
        q_i={'x': tensor([0.0])},
        p_i={'x': tensor([1.0])},
        q_f={'x': torch.sin(tensor([1.0]))},
        p_f={'x': torch.cos(tensor([1.0]))},
        prec=1e-4
    )
])
class HarmonicOscillator(object):
    @staticmethod
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * q['x'] ** 2

    @staticmethod
    def potential_fn(q):
        return 0.5 * q['x'] ** 2


@register_model([
    ModelArgs(
        step_size=0.01,
        num_steps=628,
        q_i={'x': tensor([1.0]), 'y': tensor([0.0])},
        p_i={'x': tensor([0.0]), 'y': tensor([1.0])},
        q_f={'x': tensor([1.0]), 'y': tensor([0.0])},
        p_f={'x': tensor([0.0]), 'y': tensor([1.0])},
        prec=5.0e-3
    )
])
class CircularPlanetaryMotion(object):
    @staticmethod
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * p['y'] ** 2 - \
               1.0 / torch.pow(q['x'] ** 2 + q['y'] ** 2, 0.5)

    @staticmethod
    def potential_fn(q):
        return - 1.0 / torch.pow(q['x'] ** 2 + q['y'] ** 2, 0.5)


@register_model([
    ModelArgs(
        step_size=0.1,
        num_steps=1810,
        q_i={'x': tensor([0.02])},
        p_i={'x': tensor([0.0])},
        q_f={'x': tensor([-0.02])},
        p_f={'x': tensor([0.0])},
        prec=1.0e-4
    )
])
class QuarticOscillator(object):
    @staticmethod
    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.25 * torch.pow(q['x'], 4.0)

    @staticmethod
    def potential_fn(q):
        return 0.25 * torch.pow(q['x'], 4.0)


@pytest.mark.parametrize('example', TEST_EXAMPLES, ids=EXAMPLE_IDS)
def test_trajectory(example):
    model, args = example
    q_f, p_f = velocity_verlet(args.q_i,
                               args.p_i,
                               model.potential_fn,
                               args.step_size,
                               args.num_steps)
    logger.info("initial q: {}".format(args.q_i))
    logger.info("final q: {}".format(q_f))
    assert_equal(q_f, args.q_f, args.prec)
    assert_equal(p_f, args.p_f, args.prec)


@pytest.mark.parametrize('example', TEST_EXAMPLES, ids=EXAMPLE_IDS)
def test_energy_conservation(example):
    model, args = example
    q_f, p_f = velocity_verlet(args.q_i,
                               args.p_i,
                               model.potential_fn,
                               args.step_size,
                               args.num_steps)
    energy_initial = model.energy(args.q_i, args.p_i)
    energy_final = model.energy(q_f, p_f)
    logger.info("initial energy: {}".format(energy_initial.data[0]))
    logger.info("final energy: {}".format(energy_final.data[0]))
    assert_equal(energy_final, energy_initial)


@pytest.mark.parametrize('example', TEST_EXAMPLES, ids=EXAMPLE_IDS)
def test_time_reversibility(example):
    model, args = example
    q_forward, p_forward = velocity_verlet(args.q_i,
                                           args.p_i,
                                           model.potential_fn,
                                           args.step_size,
                                           args.num_steps)
    p_reverse = {key: -val for key, val in p_forward.items()}
    q_f, p_f = velocity_verlet(q_forward,
                               p_reverse,
                               model.potential_fn,
                               args.step_size,
                               args.num_steps)
    assert_equal(q_f, args.q_i, 1e-5)
