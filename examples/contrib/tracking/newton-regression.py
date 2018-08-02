from __future__ import absolute_import, division, print_function
import math
import argparse
import os
import torch
import pdb
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from pyro.contrib.tracking.hashing import merge_points
from pyro.ops.newton import newton_step
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import ClippedAdam
from pyro.optim.multi import MixedMultiOptimizer, Newton
from pyro.util import warn_if_nan

from datagen_utils import generate_observations, get_positions
from plot_utils import plot_solution, plot_exists_prob, init_visdom

import pytest
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)


@poutine.broadcast
def model(args, observations):
    emission_noise_scale = pyro.param("emission_noise_scale")
    states_loc = pyro.param("states_loc")
    max_num_objects = 1
    with pyro.iarange("objects", max_num_objects):
        states_loc = pyro.sample("states", dist.Normal(0., 1.).expand([2]).independent(1))
        positions = get_positions(states_loc, args.num_frames)

    with pyro.iarange("detections", observations.shape[1]):
        with pyro.iarange("time", args.num_frames):
            # The remaining continuous part is exact.
            observed_positions = observations[..., 0]
            pyro.sample('observations',
                        dist.Normal(positions, emission_noise_scale),
                        obs=observed_positions)


@poutine.broadcast
def guide(args, observations):
    states_loc = pyro.param("states_loc")
    with pyro.iarange("objects", states_loc.shape[0]):
        pyro.sample("states", dist.Delta(states_loc, event_dim=1))


def init_params():
    emission_noise_scale = pyro.param("emission_noise_scale", torch.tensor(1.),
                                      constraint=constraints.positive)
    states_loc = pyro.param("states_loc", dist.Normal(0, 1).sample((args.max_num_objects, 2)))
    return states_loc, emission_noise_scale


def main(args):
    if isinstance(args, str):
        args = parse_args(args)

    # initialization
    viz = init_visdom(args.visdom)
    pyro.set_rng_seed(0)
    true_states, true_positions, observations = generate_observations(args)
    true_num_objects = len(true_states)
    max_num_detections = observations.shape[1]
    assert true_states.shape == (true_num_objects, 2)
    assert true_positions.shape == (args.num_frames, true_num_objects)
    assert observations.shape == (args.num_frames, max_num_detections, 2)
    print("generated {:d} detections from {:d} objects".format(
        (observations[..., -1] > 0).long().sum(), true_num_objects))
    print('true_states = {}'.format(true_states))

    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    init_params()

    # Run guide once and plot
    with torch.no_grad():
        states_loc = pyro.param("states_loc")
        positions = get_positions(states_loc, args.num_frames)
        p_exists = torch.ones(states_loc.shape[0])
        if viz is not None:
            plot_solution(observations, p_exists,
                          positions, true_positions, args,
                          pyro.param("emission_noise_scale").item(), 'Before inference', viz=viz)
            plot_exists_prob(p_exists, viz)

    # Optimization
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    losses = []
    ens = []

    elbo = TraceEnum_ELBO(max_iarange_nesting=2, strict_enumeration_warning=False)
    newton = Newton(trust_radii={'states_loc': 1.0})
    adam = ClippedAdam({'lr': 0.1})
    optim = MixedMultiOptimizer([(['emission_noise_scale'], adam),
                                 (['states_loc'], newton)])
    for svi_step in range(args.svi_iters):
        with poutine.trace(param_only=True) as param_capture:
            loss = elbo.differentiable_loss(model, guide, args, observations)
        params = {name: pyro.param(name).unconstrained()
                  for name in param_capture.trace.nodes.keys()}
        optim.step(loss, params)

        ens.append(pyro.param("emission_noise_scale").item())
        losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
        print('epoch {: >3d} loss = {}, emission_noise_scale = {}'.format(
            svi_step, loss, ens[-1]))

    # run visualizations
    if viz is not None:
        viz.line(losses, opts=dict(title='Loss'))
        viz.line(ens, opts=dict(title='emission_noise_scale'))

    # Run guide once and plot final result
    with torch.no_grad():
        positions = get_positions(states_loc, args.num_frames)
        p_exists = torch.ones(states_loc.shape[0])
        if viz is not None:
            plot_solution(observations, p_exists,
                          positions, true_positions, args,
                          pyro.param("emission_noise_scale").item(), 'Before inference', viz=viz)
            plot_exists_prob(p_exists, viz)


def parse_args(*args):
    from shlex import split
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-frames', default=2, type=int, help='number of frames')
    parser.add_argument('--max-num-objects', default=1, type=int, help='maximum number of objects')
    parser.add_argument('--expected-num-objects', default=1.0, type=float, help='expected number of objects')
    parser.add_argument('--expected-num-spurious', default=1e-5, type=float,
                        help='expected number of false positives, if this is too small, BP will be unstable.')
    parser.add_argument('--emission-prob', default=.9999, type=float,
                        help='emission probability, if this is too large, BP will be unstable.')
    parser.add_argument('--emission-noise-scale', default=0.1, type=float,
                        help='emission noise scale, if this is too small, SVI will see flat gradients.')
    parser.add_argument('--svi-iters', default=20, type=int, help='number of SVI iterations')
    parser.add_argument('--em-iters', default=10, type=int, help='number of EM iterations')
    parser.add_argument('--use-multi-opt', action="store_true", dest='use_multi_opt', default=False,
                        help='Whether use MixedMultiOptimizer')
    parser.add_argument('--no-visdom', action="store_false", dest='visdom', default=True,
                        help='Whether plotting in visdom is desired')
    if len(args):
        return parser.parse_args(split(args[0]))
    return parser.parse_args()


@pytest.mark.parametrize("args", ['--no-visdom'])
def test_data_generation(args):
    if isinstance(args, str):
        args = parse_args(args)
    pyro.set_rng_seed(0)
    true_states, true_positions, observations = generate_observations(args)
    true_num_objects = len(true_states)
    max_num_detections = observations.shape[1]
    assert true_states.shape == (true_num_objects, 2), \
        "true_states.shape: {}".format(true_states.shape)
    assert true_positions.shape == (args.num_frames, true_num_objects), \
        "true_positions.shape: {}".format(true_positions.shape)
    assert observations.shape == (args.num_frames, max_num_detections, 2), \
        "observations.shape: {}".format(observations.shape)


@pytest.mark.parametrize("args", ['--no-visdom'])
def test_guide(args):
    if isinstance(args, str):
        args = parse_args(args)
    pyro.set_rng_seed(0)
    true_states, true_positions, observations = generate_observations(args)
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    assignment, states_loc = guide(args, observations)
    p_exists = assignment.exists_dist.probs
    positions = get_positions(states_loc, args.num_frames)
    assert p_exists.dim() == 1
    assert positions.shape[0] == args.num_frames


@pytest.mark.parametrize("args", ['--no-visdom --svi-iters 2'])
def test_svi(args):
    if isinstance(args, str):
        args = parse_args(args)
    pyro.set_rng_seed(0)
    true_states, true_positions, observations = generate_observations(args)
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()

    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    optim = ClippedAdam({'lr': 0.1})
    svi = SVI(model, guide, optim, elbo)
    losses = []
    for epoch in range(args.svi_iters):
        loss = svi.step(args, observations)
        losses.append(loss)
        print('epoch {: >3d} loss = {}, emission_noise_scale = {}'.format(
            epoch, loss, pyro.param("emission_noise_scale").item()))


if __name__ == '__main__':
    args = parse_args()
    assert args.max_num_objects >= args.expected_num_objects
    main(args)
