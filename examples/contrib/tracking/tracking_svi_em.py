from __future__ import absolute_import, division, print_function
import math
import os
import torch
from torch.distributions import constraints
from matplotlib import pyplot
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from pyro.contrib.tracking.hashing import LSH, merge_points
from pyro.ops.newton import newton_step
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import ClippedAdam, ASGD, SGD
from pyro.util import warn_if_nan

from datagen_utils import generate_observations, get_positions
from plot_utils import plot_solution, plot_exists_prob, init_visdom
pyro.enable_validation(True)
import pytest
smoke_test = ('CI' in os.environ)
import argparse


@poutine.broadcast
def model(args, observations):
    emission_noise_scale = pyro.param("emission_noise_scale", torch.tensor(10.1234),
                                      constraint=constraints.positive)
    max_num_objects = pyro.sample("max_num_objects",
                                  dist.Geometric(1. / args.max_num_objects)).long().item()
    with pyro.iarange("objects", max_num_objects):
        exists = pyro.sample("exists",
                             dist.Bernoulli(args.expected_num_objects / max_num_objects))
        with poutine.scale(scale=exists):
            states_loc = pyro.sample("states", dist.Normal(0., 1.).expand([2]).independent(1))
            positions = get_positions(states_loc, args.num_frames)
    with pyro.iarange("detections", observations.shape[1]):
        with pyro.iarange("time", args.num_frames):
            # The combinatorial part of the log prob is approximated to allow independence.
            is_observed = (observations[..., -1] > 0)
            with poutine.scale(scale=is_observed.float()):
                assign = pyro.sample("assign",
                                     dist.Categorical(torch.ones(max_num_objects + 1)))
            is_spurious = (assign == max_num_objects)
            is_real = is_observed & ~is_spurious
            num_observed = is_observed.float().sum(-1, True)
            # TODO Make these Bernoulli probs more plausible.
            pyro.sample("is_real",
                        dist.Bernoulli(args.expected_num_objects / max_num_objects),
                        obs=is_real.float())
            pyro.sample("is_spurious",
                        dist.Bernoulli(args.expected_num_spurious / max_num_objects),
                        obs=is_spurious.float())

            # The remaining continuous part is exact.
            observed_positions = observations[..., 0]
            print(observed_positions.shape,is_real.shape)
            with poutine.scale(scale=is_real.float()):
                bogus_position = positions.new_zeros(args.num_frames, 1)
                augmented_positions = torch.cat([positions, bogus_position], -1)
                predicted_positions = augmented_positions[:, assign]
                pyro.sample("real_observations",
                            dist.Normal(predicted_positions, emission_noise_scale),
                            obs=observed_positions)
            with poutine.scale(scale=is_spurious.float()):
                pyro.sample("spurious_observations", dist.Normal(0., 1.),
                            obs=observed_positions)


def compute_exists_logits(states_loc, replicates):
    FUDGE = -5
    return states_loc.new_empty(states_loc.shape[0]).fill_(-math.log(replicates) + FUDGE)


def compute_assign_logits(positions, observations, replicates, args):
    log_likelihood = detection_log_likelihood(positions, observations, args)
    assign_logits = log_likelihood[..., :-1] - log_likelihood[..., -1:] - math.log(replicates)
    is_observed = (observations[..., -1] > 0)
    assign_logits[~is_observed] = -float('inf')
    return assign_logits


def detection_log_likelihood(positions, observations, args):
    real_dist = dist.Normal(positions.unsqueeze(-2), args.emission_noise_scale)
    spurious_dist = dist.Normal(0., 1.)
    is_observed = (observations[..., -1] > 0)
    observed_positions = observations[..., 0].unsqueeze(-1)
    a = (real_dist.log_prob(observed_positions) +
         math.log(args.expected_num_objects * args.emission_prob))
    b = (spurious_dist.log_prob(observed_positions) +
         math.log(args.expected_num_spurious))
    log_likelihood = torch.cat((a, b), dim=-1)
    log_likelihood[~is_observed] = -float('inf')
    return log_likelihood


@poutine.broadcast
def guide(args, observations):
    # Initialize states randomly from the prior.
    # states_loc = torch.randn(args.max_num_objects, 2)
    # DEBUG this attempts to initialize to truth
    states_loc = torch.tensor([[1.5410, -0.2934], [-2.1788, 0.5684]] +
                              [[0., 0.]] * (args.max_num_objects - 2))
    is_observed = (observations[..., -1] > 0)

    for em_iter in range(args.em_iters):
        states_loc = states_loc.detach()
        states_loc.requires_grad = True
        positions = get_positions(states_loc, args.num_frames)
        replicates = max(1, states_loc.shape[0] / args.expected_num_objects)
        # E-step: compute soft assignments
        with torch.set_grad_enabled(False):
            assign_logits = compute_assign_logits(positions, observations, replicates, args)
            exists_logits = compute_exists_logits(states_loc, replicates)
            assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                                      args.bp_iters,
                                                      bp_momentum=args.bp_momentum)
            p_exists = assignment.exists_dist.probs
            p_assign = assignment.assign_dist.probs

        log_likelihood = detection_log_likelihood(positions, observations, args)
        loss = -(log_likelihood * p_assign).sum()
        # M-step:
        states_loc, _ = newton_step(loss, states_loc, args.emission_noise_scale)

        if args.prune_threshold > 0.0:
            states_loc = states_loc[p_exists > args.prune_threshold]
        if args.merge_radius >= 0.0:
            states_loc, _ = merge_points(states_loc, args.merge_radius)

        warn_if_nan(states_loc, 'states_loc')

    positions = get_positions(states_loc, args.num_frames)
    replicates = max(1, states_loc.shape[0] / args.expected_num_objects)
    assign_logits = compute_assign_logits(positions, observations, replicates, args)
    exists_logits = compute_exists_logits(states_loc, replicates)
    assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                              args.bp_iters, bp_momentum=args.bp_momentum)
    pyro.sample("max_num_objects", dist.Delta(torch.tensor(float(len(states_loc)))))
    with pyro.iarange("objects", states_loc.shape[0]):
        exists = pyro.sample("exists", assignment.exists_dist, infer={"enumerate": "parallel"})
        with poutine.scale(scale=exists):
            #states_var = states_cov.reshape(states_cov.shape[:-2] + (-1,))[..., ::n+1]
            #pyro.sample("states", dist.Normal(states_loc, states_var).independent(1))
            pyro.sample("states", dist.Delta(states_loc, event_dim=1))
    with pyro.iarange("detections", observations.shape[1]):
        with poutine.scale(scale=is_observed.float()):
            with pyro.iarange("time", args.num_frames):
                pyro.sample("assign", assignment.assign_dist, infer={"enumerate": "parallel"})

    return assignment, states_loc


def main(args):
    if isinstance(obj, str):
        args = make_args(args)

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
    assignment, states_loc = guide(args, observations)
    p_exists = assignment.exists_dist.probs
    positions = get_positions(states_loc, args.num_frames)
    plot_solution(observations, p_exists, positions, true_positions, args, 'after 10 EM (with merge)', viz=viz)
    plot_exists_prob(p_exists, viz)

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
    if viz is not None:
        viz.plot(losses)
    else:
        pyplot.figure().patch.set_color('white')
        pyplot.plot(losses)

    assignment, states_loc = guide(args, observations)
    p_exists = assignment.exists_dist.probs
    positions = get_positions(states_loc, args.num_frames)
    plot_solution(observations, p_exists, positions, true_positions, args,
                  'after 10 EM (with prune and merge)', viz=viz)
    plot_exists_prob(p_exists, viz)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_frames', default=40, type=int, help='number of frames')
    parser.add_argument('--max_num_objects', default=400, type=int, help='maximum number of objects')
    parser.add_argument('--expected_num_objects', default=2.0, type=float, help='expected number of objects')
    parser.add_argument('--expected_num_spurious', default=0.2, type=float,
                        help='expected number of false positives, if this is too small, BP will be unstable.')
    parser.add_argument('--emission_prob', default=0.8, type=float,
                        help='emission probability, if this is too large, BP will be unstable.')
    parser.add_argument('--emission_noise_scale', default=0.1, type=float,
                        help='emission noise scale, if this is too small, SVI will see flat gradients.')
    parser.add_argument('--bp_iters', default=50, type=int, help='number of BP iterations')
    parser.add_argument('--bp_momentum', default=0.5, type=float, help='BP momentum')
    parser.add_argument('--svi_iters', default=20, type=int, help='number of SVI iterations')
    parser.add_argument('--em_iters', default=10, type=int, help='number of EM iterations')
    parser.add_argument('--merge_radius', default=0.5, type=float, help='merge radius')
    parser.add_argument('--prune_threshold', default=1e-2, type=float, help='prune threshold')
    parser.add_argument('--no_visdom', action="store_false", dest='visdom', default=True,
                        help='Whether plotting in visdom is desired')
    return parser


def make_args(args_string):
    from shlex import split
    return arg_parser().parse_args(split(args_string))


@pytest.mark.parametrize("args", ['--no_visdom'])
def test_data_generation(args):
    if isinstance(args, str):
        args = make_args(args)
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


@pytest.mark.parametrize("args", ['--no_visdom'])
def test_guide(args):
    if isinstance(args, str):
        args = make_args(args)
    pyro.set_rng_seed(0)
    true_states, true_positions, observations = generate_observations(args)
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    assignment, states_loc = guide(args, observations)
    p_exists = assignment.exists_dist.probs
    positions = get_positions(states_loc, args.num_frames)


@pytest.mark.parametrize("args", ['--no_visdom --svi_iters 2'])
def test_svi(args):
    if isinstance(args, str):
        args = make_args(args)

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
    args = arg_parser().parse_args()
    assert args.max_num_objects >= args.expected_num_objects
    main(args)
