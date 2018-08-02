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
    max_num_objects = 2
    with pyro.iarange("objects", max_num_objects):
        states_loc = pyro.sample("states", dist.Normal(0., 1.).expand([2]).independent(1))
        positions = get_positions(states_loc, args.num_frames)

    with pyro.iarange("detections", observations.shape[1]):
        with pyro.iarange("time", args.num_frames):
            # The remaining continuous part is exact.
            assign = pyro.sample("assign",
                                 dist.Categorical(torch.ones(max_num_objects + 1)))
            observed_positions = observations[..., 0]
            bogus_position = positions.new_zeros(args.num_frames, 1)
            augmented_positions = torch.cat([positions, bogus_position], -1)
            # weird tricks because index and input must be same dimension in gather
            assign = torch.cat((assign, assign[..., :1]), -1)
            predicted_positions = torch.gather(augmented_positions, -1, assign)
            predicted_positions = predicted_positions[..., :-1]
            #pdb.set_trace()
            pyro.sample('observations', dist.Normal(predicted_positions, emission_noise_scale),
                        obs=observed_positions)


@poutine.broadcast
def guide(args, observations):
    states_loc = pyro.param("states_loc")
    # emission_noise_scale = pyro.param("emission_noise_scale")
    assign_probs = torch.zeros(args.num_frames, observations.shape[1], states_loc.shape[0] + 1)
    assign_probs[:, 0, 0] = 0.99
    assign_probs[:, 0, 1] = 0.01
    assign_probs[:, 1, 1] = 0.99
    assign_probs[:, 1, 0] = 0.01
    assign_dist = dist.Categorical(probs=assign_probs)
    with pyro.iarange("objects", states_loc.shape[0]):
            #  states_var = states_cov.reshape(states_cov.shape[:-2] + (-1,))[..., ::n+1]
            #  pyro.sample("states", dist.Normal(states_loc, states_var).independent(1))
        pyro.sample("states", dist.Delta(states_loc, event_dim=1))

    with pyro.iarange("detections", observations.shape[1]):
        with pyro.iarange("time", args.num_frames):
            pyro.sample("assign", assign_dist, infer={"enumerate": "sequential"})


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
    pyro.set_rng_seed(args.seed)
    true_states, true_positions, observations = generate_observations(args)
    true_num_objects = len(true_states)
    max_num_detections = observations.shape[1]
    assert true_states.shape == (true_num_objects, 2)
    assert true_positions.shape == (args.num_frames, true_num_objects)
    assert observations.shape == (args.num_frames, max_num_detections, 2)
    print("generated {:d} detections from {:d} objects".format(
        (observations[..., -1] > 0).long().sum(), true_num_objects))
    print('true_states = {}'.format(true_states))

    pyro.set_rng_seed(args.seed + 1)  # Use a different seed from data generation
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
    pyro.set_rng_seed(args.seed + 1)  # Use a different seed from data generation
    losses = []
    ens = []

    elbo = TraceEnum_ELBO(max_iarange_nesting=2, strict_enumeration_warning=False)
    newton = Newton(trust_radii={'states_loc': 1.0})
    adam = ClippedAdam({'lr': 0.1})
    optim = MixedMultiOptimizer([(['emission_noise_scale'], adam),
                                 (['states_loc'], newton)])
    try:
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
    except KeyboardInterrupt:
        print('Interrupted')

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
    parser.add_argument('--num-frames', default=10, type=int, help='number of frames')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--max-num-objects', default=2, type=int, help='maximum number of objects')
    parser.add_argument('--expected-num-objects', default=2.0, type=float, help='expected number of objects')
    parser.add_argument('--expected-num-spurious', default=1e-5, type=float,
                        help='expected number of false positives, if this is too small, BP will be unstable.')
    parser.add_argument('--emission-prob', default=.9999, type=float,
                        help='emission probability, if this is too large, BP will be unstable.')
    parser.add_argument('--emission-noise-scale', default=0.1, type=float,
                        help='emission noise scale, if this is too small, SVI will see flat gradients.')
    parser.add_argument('--svi-iters', default=20, type=int, help='number of SVI iterations')
    parser.add_argument('--no-visdom', action="store_false", dest='visdom', default=True,
                        help='Whether plotting in visdom is desired')
    if len(args):
        return parser.parse_args(split(args[0]))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.max_num_objects >= args.expected_num_objects
    main(args)
