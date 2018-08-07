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
    max_num_objects = pyro.sample("max_num_objects",
                                  dist.Geometric(1. / args.max_num_objects)).long().item()
    with pyro.iarange("objects", max_num_objects):
        exists = pyro.sample("exists",
                             dist.Bernoulli(min(1., args.expected_num_objects / max_num_objects)))
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

            # TODO Make these Bernoulli probs more plausible.
            pyro.sample("is_real",
                        dist.Bernoulli(min(0.999, args.expected_num_objects / max_num_objects)),
                        obs=is_real.float())
            pyro.sample("is_spurious",
                        dist.Bernoulli(min(0.999, args.expected_num_spurious / max_num_objects)),
                        obs=is_spurious.float())

            # The remaining continuous part is exact.
            observed_positions = observations[..., 0]
            bogus_position = positions.new_zeros(args.num_frames, 1)
            augmented_positions = torch.cat([positions, bogus_position], -1)
            # weird tricks because index and input must be same dimension in gather
            pad_shape = assign.shape[:-1] + (augmented_positions.shape[-1] - assign.shape[-1],)
            assign = torch.cat(
                (assign,
                 torch.zeros(assign[..., :1].shape, dtype=torch.long).expand(pad_shape)
                ), -1)
            augmented_positions = augmented_positions.unsqueeze(0).expand_as(assign)
            predicted_positions = torch.gather(augmented_positions, -1, assign)
            pyro.sample('observations', dist.MaskedMixture(is_real,
                                                           dist.Normal(0., 1.),  # fake dist
                                                           dist.Normal(predicted_positions,  # real dist
                                                                       emission_noise_scale)
                                                           ),
                        obs=observed_positions)


def compute_exists_logits(states_loc, args):
    replicates = max(1, states_loc.shape[0] / args.expected_num_objects)
    log_likelihood = exists_log_likelihood(states_loc, args)
    exists_logits = log_likelihood[:, 0] - log_likelihood[:, 1] - math.log(replicates)
    return exists_logits


def exists_log_likelihood(states_loc, args):
    p_exists = min(0.9999, args.expected_num_objects / states_loc.shape[0])
    real_part = dist.Normal(0., 1.).log_prob(states_loc).sum(-1)
    real_part = real_part + math.log(p_exists)
    spurious_part = torch.empty(real_part.shape).fill_(math.log(1 - p_exists))
    return torch.stack([real_part, spurious_part], -1)


def compute_assign_logits(positions, observations, emission_noise_scale, args):
    replicates = max(1, positions.shape[1] / args.expected_num_objects)
    log_likelihood = assign_log_likelihood(positions, observations, emission_noise_scale, args)
    assign_logits = log_likelihood[..., :-1] - log_likelihood[..., -1:] - math.log(replicates)
    is_observed = (observations[..., -1] > 0)
    assign_logits[~is_observed] = -float('inf')
    return assign_logits


def assign_log_likelihood(positions, observations, emission_noise_scale, args):
    real_dist = dist.Normal(positions.unsqueeze(-2), args.emission_noise_scale)
    fake_dist = dist.Uniform(-3., 3.)
    is_observed = (observations[..., -1] > 0)
    observed_positions = observations[..., 0].unsqueeze(-1)
    p_real = min(0.999, args.expected_num_objects / observations.shape[1])
    p_fake = min(0.999, args.expected_num_spurious / observations.shape[1])
    real_part = real_dist.log_prob(observed_positions) + math.log(p_real)
    fake_part = fake_dist.log_prob(observed_positions) + math.log(p_fake)
    log_likelihood = torch.cat([real_part, fake_part], -1)
    log_likelihood[~is_observed] = -float('inf')
    return log_likelihood


@poutine.broadcast
def guide(args, observations):
    emission_noise_scale = pyro.param("emission_noise_scale")
    states_loc = pyro.param("states_loc")
    is_observed = (observations[..., -1] > 0)
    # states_loc = states_loc.detach()
    # states_loc.requires_grad = True
    positions = get_positions(states_loc, args.num_frames)
    with torch.set_grad_enabled(True):
        assign_logits = compute_assign_logits(positions, observations,
                                              emission_noise_scale, args)
        exists_logits = compute_exists_logits(states_loc, args)
        assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                                  bp_iters=args.bp_iters, bp_momentum=args.bp_momentum)

    pyro.sample("max_num_objects", dist.Delta(torch.tensor(float(states_loc.shape[0]))))
    with pyro.iarange("objects", states_loc.shape[0]):
        exists = pyro.sample("exists", assignment.exists_dist, infer={"enumerate": "parallel"})
        with poutine.scale(scale=exists):
            #  states_var = states_cov.reshape(states_cov.shape[:-2] + (-1,))[..., ::n+1]
            #  pyro.sample("states", dist.Normal(states_loc, states_var).independent(1))
            pyro.sample("states", dist.Delta(states_loc, event_dim=1))
    with poutine.scale(scale=is_observed.float()):
        with pyro.iarange("detections", observations.shape[1]):
            with pyro.iarange("time", args.num_frames):
                pyro.sample("assign", assignment.assign_dist, infer={"enumerate": "parallel"})

    return assignment, states_loc


def init_params():
    emission_noise_scale = pyro.param("emission_noise_scale", torch.tensor(.5),
                                      constraint=constraints.positive)
    states_loc = pyro.param("states_loc", dist.Normal(0., 1.).sample((args.max_num_objects, 2)))
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
        assignment, states_loc = guide(args, observations)
        p_exists = assignment.exists_dist.probs
        positions = get_positions(states_loc, args.num_frames)
        if viz is not None:
            plot_solution(observations, p_exists, positions, true_positions, args,
                          pyro.param("emission_noise_scale").item(), 'Before inference', viz=viz)
            plot_exists_prob(p_exists, viz)

    # Optimization
    pyro.clear_param_store()
    init_params()
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    losses = []
    ens = []

    # Learn states_loc via EM and emission_noise_scale via SVI.

    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    newton = Newton(trust_radii={'states_loc': 1.0})
    if args.use_multi_opt:
        adam = ClippedAdam({'lr': 0.1})
        optim = MixedMultiOptimizer([(['emission_noise_scale'], adam),
                                     (['states_loc'], newton)])
    else:
        optim = ClippedAdam({'lr': 0.1})
        svi = SVI(poutine.block(model, hide=['states_loc']),
                  poutine.block(guide, hide=['states_loc']), optim, elbo)

    for svi_step in range(args.svi_iters):
        if args.use_multi_opt:
            with poutine.trace(param_only=True) as param_capture:
                loss = elbo.differentiable_loss(model, guide, args, observations)
            params = {name: pyro.param(name).unconstrained()
                      for name in param_capture.trace.nodes.keys()}
            optim.step(loss, params)
        else:
            for em_step in range(args.em_iters):
                states_loc = pyro.param('states_loc').detach_().requires_grad_()
                assert pyro.param('states_loc').grad_fn is None
                loss = elbo.differentiable_loss(model, guide, args, observations) # + 100 * pyro.param("emission_noise_scale").pow(2)  # E-step
                updated = newton.get_step(loss, {'states_loc': states_loc})  # M-step
                updated_states_loc = updated['states_loc']
                assert pyro.param('states_loc').grad_fn is not None
            loss = svi.step(args, observations)
        with torch.no_grad():
            assignment, _ = guide(args, observations)
            p_exists = assignment.exists_dist.probs
            updated_states_loc = pyro.param("states_loc")
            if args.prune_threshold > 0.0:
                updated_states_loc = updated_states_loc[p_exists > args.prune_threshold]
            if (args.merge_radius >= 0.0) and updated_states_loc.dim() == 2:
                updated_states_loc, _ = merge_points(updated_states_loc, args.merge_radius)
            #assert updated_states_loc.grad_fn is not None
            pyro.get_param_store().replace_param('states_loc', updated_states_loc, pyro.param("states_loc"))

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
        assignment, states_loc = guide(args, observations)
        p_exists = assignment.exists_dist.probs
        positions = get_positions(states_loc, args.num_frames)
        if viz is not None:
            plot_solution(observations, p_exists, positions, true_positions, args,
                          pyro.param("emission_noise_scale").item(),
                          'After inference', viz=viz)
            plot_exists_prob(p_exists, viz)


def parse_args(*args):
    from shlex import split
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-frames', default=40, type=int, help='number of frames')
    parser.add_argument('--max-num-objects', default=400, type=int, help='maximum number of objects')
    parser.add_argument('--expected-num-objects', default=2.0, type=float, help='expected number of objects')
    parser.add_argument('--expected-num-spurious', default=0.2, type=float,
                        help='expected number of false positives, if this is too small, BP will be unstable.')
    parser.add_argument('--emission-prob', default=0.8, type=float,
                        help='emission probability, if this is too large, BP will be unstable.')
    parser.add_argument('--emission-noise-scale', default=0.1, type=float,
                        help='emission noise scale, if this is too small, SVI will see flat gradients.')
    parser.add_argument('--bp-iters', default=50, type=int, help='number of BP iterations')
    parser.add_argument('--bp-momentum', default=0.5, type=float, help='BP momentum')
    parser.add_argument('--svi-iters', default=20, type=int, help='number of SVI iterations')
    parser.add_argument('--em-iters', default=10, type=int, help='number of EM iterations')
    parser.add_argument('--merge-radius', default=0.5, type=float, help='merge radius')
    parser.add_argument('--prune-threshold', default=1e-2, type=float, help='prune threshold')
    parser.add_argument('--use-multi-opt', action="store_true", dest='use_multi_opt', default=False,
                        help='Whether use MixedMultiOptimizer')
    parser.add_argument('--no-visdom', action="store_false", dest='visdom', default=True,
                        help='Whether plotting in visdom is desired')
    if len(args):
        return parser.parse_args(split(args[0]))
    args = parser.parse_args()
    if args.bp_iters < 0:
        args.bp_iters = None
    return args


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
