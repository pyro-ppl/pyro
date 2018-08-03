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
                                 dist.Categorical(torch.ones(args.max_num_objects + 1)))
            observed_positions = observations[..., 0]
            bogus_position = positions.new_zeros(args.num_frames, 1)
            augmented_positions = torch.cat([positions, bogus_position], -1)
            #predicted_positions = augmented_positions[:, assign]
            # weird tricks because index and input must be same dimension in gather
            pad_shape = assign.shape[:-1] + (augmented_positions.shape[-1] - assign.shape[-1],)
            assign = torch.cat(
                (assign,
                 torch.zeros(assign[..., :1].shape, dtype=torch.long).expand(pad_shape)
                ), -1)
            augmented_positions = augmented_positions.unsqueeze(0).expand_as(assign)
            predicted_positions = torch.gather(augmented_positions, -1, assign)
            predicted_positions = predicted_positions[..., :-1]
            if args.debug: pdb.set_trace()
            pyro.sample('observations', dist.Normal(predicted_positions, emission_noise_scale),
                        obs=observed_positions)


def compute_exists_logits(states_loc, args):
    log_likelihood = exists_log_likelihood(states_loc, args)
    exists_logits = log_likelihood[:, 0] - log_likelihood[:, 1]
    return exists_logits


def exists_log_likelihood(states_loc, args):
    p_exists = min(0.9999, args.expected_num_objects / states_loc.shape[0])
    real_part = torch.empty(states_loc.shape[0]).fill_(math.log(p_exists))
    spurious_part = torch.empty(real_part.shape).fill_(math.log(1 - p_exists))
    return torch.stack([real_part, spurious_part], -1)


def compute_assign_logits(positions, observations, emission_noise_scale, args):
    log_likelihood = assign_log_likelihood(positions, observations, emission_noise_scale, args)
    assign_logits = log_likelihood[..., :-1] - log_likelihood[..., -1:]
    is_observed = (observations[..., -1] > 0)
    assign_logits[~is_observed] = -float('inf')
    return assign_logits


def assign_log_likelihood(positions, observations, emission_noise_scale, args):
    real_dist = dist.Normal(positions.unsqueeze(-2), emission_noise_scale)
    fake_dist = dist.Uniform(-4., 4.)
    is_observed = (observations[..., -1] > 0)
    observed_positions = observations[..., :-1]
    real_part = real_dist.log_prob(observed_positions)
    fake_part = fake_dist.log_prob(observed_positions)
    log_likelihood = torch.cat([real_part, fake_part], -1)
    log_likelihood[~is_observed] = -float('inf')
    return log_likelihood


@poutine.broadcast
def guide(args, observations):
    states_loc = pyro.param("states_loc")
    emission_noise_scale = pyro.param("emission_noise_scale")
    with pyro.iarange("objects", states_loc.shape[0]):
        pyro.sample("states", dist.Delta(states_loc, event_dim=1))
    #pdb.set_trace()
    positions = get_positions(states_loc, args.num_frames)
    assign_logits = compute_assign_logits(positions, observations,
                                          emission_noise_scale, args)
    exists_logits = compute_exists_logits(states_loc, args)
    assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                              bp_iters=args.bp_iters, bp_momentum=args.bp_momentum)
    if args.debug: pdb.set_trace()
    #assign_dist = dist.Categorical(logits=assign_logits)
    assign_dist = assignment.assign_dist
    with pyro.iarange("detections", observations.shape[1]):
        with pyro.iarange("time", args.num_frames):
            pyro.sample("assign", assign_dist, infer={"enumerate": "parallel"})


def init_params(true_states=None):
    emission_noise_scale = pyro.param("emission_noise_scale", torch.tensor(1.),
                                      constraint=constraints.positive)
    if true_states is not None:
        states_loc = pyro.param("states_loc", lambda: true_states)
    else:
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
    init_params(dist.Normal(true_states, .5).sample())

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
    newton = Newton(trust_radii={'states_loc': .5})
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
            if args.debug: print(pyro.param("states_loc"))
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
                          pyro.param("emission_noise_scale").item(), 'After inference', viz=viz)
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
    parser.add_argument('--bp-iters', default=20, type=int, help='number of BP iterations')
    parser.add_argument('--bp-momentum', default=0.5, type=float, help='BP momentum')
    parser.add_argument('--no-visdom', action="store_false", dest='visdom', default=True,
                        help='Whether plotting in visdom is desired')
    parser.add_argument('--debug', action="store_true", dest='debug', default=False,
                        help='Whether plotting in visdom is desired')
    if len(args):
        return parser.parse_args(split(args[0]))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.max_num_objects >= args.expected_num_objects
    main(args)
