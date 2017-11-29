from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import torch
from torch.autograd import Variable

from pyro.optim.integrator import velocity_verlet
from tests.common import assert_equal

logger = logging.getLogger(__name__)


def test_energy_conservation_harmonic_oscillator():

    def energy(q, p):
        return 0.5 * p['x']**2 + 0.5 * q['x']**2

    def potential_fn(q):
        return 0.5 * q['x']**2

    q = {'x': Variable(torch.Tensor([0.0]))}
    p = {'x': Variable(torch.Tensor([1.0]))}
    energy_cur = energy(q, p)
    logger.info("*** harmonic oscillator ***")
    logger.info("Energy - current: {}".format(energy_cur.data[0]))
    q_new, p_new = velocity_verlet(q, p, potential_fn, 0.01, 100)
    assert q_new['x'].data[0] != q['x'].data[0]
    energy_new = energy(q_new, p_new)
    assert_equal(energy_new, energy_cur)
    assert_equal(q_new['x'].data[0], np.sin(1.0), prec=1.0e-4)
    assert_equal(p_new['x'].data[0], np.cos(1.0), prec=1.0e-4)
    logger.info("q_old: {}, p_old: {}".format(q['x'].data[0], p['x'].data[0]))
    logger.info("q_new: {}, p_new: {}".format(q_new['x'].data[0], p_new['x'].data[0]))
    logger.info("Energy - new: {}".format(energy_new.data[0]))


def test_time_reversibility_harmonic_oscillator():

    def energy(q, p):
        return 0.5 * p['x']**2 + 0.5 * q['x']**2

    def potential_fn(q):
        return 0.5 * q['x']**2

    q = {'x': Variable(torch.Tensor([0.0]))}
    p = {'x': Variable(torch.Tensor([1.0]))}
    q_forward, p_forward = velocity_verlet(q, p, potential_fn, 0.01, 100)
    p_reverse = {'x': -p_forward['x']}
    q_final, p_final = velocity_verlet(q_forward, p_reverse, potential_fn, 0.01, 100)
    assert_equal(q, q_final, 1e-5)


def test_energy_conservation_circular_planetary_motion():

    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * p['y'] ** 2 - \
            1.0 / torch.pow(q['x'] ** 2 + q['y'] ** 2, 0.5)

    def potential_fn(q):
        return - 1.0 / torch.pow(q['x'] ** 2 + q['y'] ** 2, 0.5)

    q = {'x': Variable(torch.Tensor([1.0])), 'y': Variable(torch.Tensor([0.0]))}
    p = {'x': Variable(torch.Tensor([0.0])), 'y': Variable(torch.Tensor([1.0]))}
    energy_initial = energy(q, p)
    logger.info("*** circular planetary motion ***")
    logger.info("initial energy: {}".format(energy_initial.data[0]))
    q_new, p_new = velocity_verlet(q, p, potential_fn, 0.01, 628)
    energy_final = energy(q_new, p_new)
    assert_equal(energy_final, energy_initial)
    assert_equal(q_new['x'].data[0], 1.0, prec=5.0e-3)
    assert_equal(q_new['y'].data[0], 0.0, prec=5.0e-3)
    logger.info("final energy: {}".format(energy_final.data[0]))


def test_energy_conservation_quartic_oscillator():

    def energy(q, p):
        return 0.5 * p['x']**2 + 0.25 * torch.pow(q['x'], 4.0)

    def potential_fn(q):
        return 0.25 * torch.pow(q['x'], 4.0)

    q = {'x': Variable(torch.Tensor([0.02]))}
    p = {'x': Variable(torch.Tensor([0.0]))}
    energy_initial = energy(q, p)
    logger.info("*** quartic oscillator ***")
    logger.info("initial energy: {}".format(energy_initial.data[0]))
    q_new, p_new = velocity_verlet(q, p, potential_fn, 0.1, 1810)
    energy_final = energy(q_new, p_new)
    assert_equal(energy_final, energy_initial)
    assert_equal(q_new['x'].data[0], -0.02, prec=1.0e-4)
    assert_equal(p_new['x'].data[0], 0.0, prec=1.0e-4)
    logger.info("final energy: {}".format(energy_final.data[0]))
