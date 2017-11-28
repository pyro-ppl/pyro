from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyro.infer.mcmc.verlet_integrator import verlet_integrator
from tests.common import assert_equal


def test_harmonic_oscillator():

    def energy(q, p):
        return 0.5 * p['x']**2 + 0.5 * q['x']**2

    def grad(q):
        return {'x': q['x']}

    q = {'x': Variable(torch.Tensor([0.0]), requires_grad=True)}
    p = {'x': Variable(torch.Tensor([1.0]), requires_grad=True)}
    energy_cur = energy(q, p)
    print("Energy - current: {}".format(energy_cur.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.01, 100)
    assert q_new['x'].data[0] != q['x'].data[0]
    energy_new = energy(q_new, p_new)
    assert_equal(energy_new, energy_cur)
    assert_equal(q_new['x'].data[0], np.sin(1.0), prec=1.0e-4)
    assert_equal(p_new['x'].data[0], np.cos(1.0), prec=1.0e-4)
    print("q_old: {}, p_old: {}".format(q['x'].data[0], p['x'].data[0]))
    print("q_new: {}, p_new: {}".format(q_new['x'].data[0], p_new['x'].data[0]))
    print("Energy - new: {}".format(energy_new.data[0]))
    print("-------------------------------------")


def test_circular_planetary_motion():

    def energy(q, p):
        return 0.5 * p['x'] ** 2 + 0.5 * p['y'] ** 2 - \
            1.0 / torch.pow(q['x'] ** 2 + q['y'] ** 2, 0.5)

    def grad(q):
        return {
            'x': q['x'] / torch.pow(q['x']**2 + q['y']**2, 1.5),
            'y': q['y'] / torch.pow(q['x']**2 + q['y']**2, 1.5)
        }

    q = {'x': Variable(torch.Tensor([1.0]), requires_grad=True), 'y': Variable(torch.Tensor([0.0]), requires_grad=True)}
    p = {'x': Variable(torch.Tensor([0.0]), requires_grad=True), 'y': Variable(torch.Tensor([1.0]), requires_grad=True)}
    energy_initial = energy(q, p)
    print("*** circular planetary motion ***")
    print("initial energy: {}".format(energy_initial.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.01, 628)
    energy_final = energy(q_new, p_new)
    assert_equal(energy_final, energy_initial)
    assert_equal(q_new['x'].data[0], 1.0, prec=5.0e-3)
    assert_equal(q_new['y'].data[0], 0.0, prec=5.0e-3)
    print("final energy: {}".format(energy_final.data[0]))
    print("-------------------------------------")


def test_quartic_oscillator():

    def energy(q, p):
        return 0.5 * p['x']**2 + 0.25 * torch.pow(q['x'], 4.0)

    def grad(q):
        return {'x': torch.pow(q['x'], 3.0)}

    q = {'x': Variable(torch.Tensor([0.02]), requires_grad=True)}
    p = {'x': Variable(torch.Tensor([0.0]), requires_grad=True)}
    energy_initial = energy(q, p)
    print("*** quartic oscillator ***")
    print("initial energy: {}".format(energy_initial.data[0]))
    q_new, p_new = verlet_integrator(q, p, grad, 0.1, 1810)
    energy_final = energy(q_new, p_new)
    assert_equal(energy_final, energy_initial)
    assert_equal(q_new['x'].data[0], -0.02, prec=1.0e-4)
    assert_equal(p_new['x'].data[0], 0.0, prec=1.0e-4)
    print("final energy: {}".format(energy_final.data[0]))
    print("-------------------------------------")
