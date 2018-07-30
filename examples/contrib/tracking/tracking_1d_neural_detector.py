from __future__ import absolute_import, division, print_function
import math
import os
import torch
from torch import nn
import argparse

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from pyro.contrib.tracking.hashing import merge_points
from pyro.ops.newton import newton_step
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import ClippedAdam
from pyro.util import warn_if_nan

from datagen_utils import get_positions, generate_sensor_data, raster2vector, vector2raster

import plot_utils
from matplotlib import pyplot

import pytest
pyro.enable_validation(True)
smoke_test = ('CI' in os.environ)


def plot_solution(sensor_outputs, observations, p_exists, positions,
                  true_positions, args, message='', viz=None):
    fig, ax = pyplot.subplots(figsize=(12, 9))
    fig.patch.set_color('white')
    extent = [-.5, args.num_frames - .5, args.x_min, args.x_max]
    cax = ax.matshow(sensor_outputs.t(), aspect='auto', extent=extent, origin='lower', alpha=0.5)
    pyplot.colorbar(cax)
    plot_utils.plot_solution(observations[..., :-1], p_exists, positions,
                             true_positions, args, message='', fig=fig, viz=viz)


def compute_exists_logits(states_loc, replicates):
    FUDGE = -5
    # TODO add a term for prior over object location
    return states_loc.new_empty(states_loc.shape[0]).fill_(-math.log(replicates) + FUDGE)


def compute_assign_logits(positions, observations, replicates, args):
    log_likelihood = detection_log_likelihood(positions, observations, args)
    assign_logits = log_likelihood[..., :-1] - log_likelihood[..., -1:] - math.log(replicates)
    assign_logits[log_likelihood[..., :-1] == -float('inf')] = -float('inf')
    return assign_logits


def detection_log_likelihood(positions, observations, args):
    noise_power = 10 ** (-args.PNR / 10)
    bin_width = (args.x_max - args.x_min) / args.num_sensors
    real_loc_dist = dist.Normal(positions.unsqueeze(-2), bin_width)
    real_output_dist = dist.Normal(1., noise_power)
    spurious_output_dist = dist.Normal(0., noise_power)
    spurious_loc_dist = dist.Uniform(args.x_min, args.x_max)
    observed_positions = observations[..., 0].unsqueeze(-1)
    observed_outputs = observations[..., 2].unsqueeze(-1)
    a = (real_loc_dist.log_prob(observed_positions) +
         real_output_dist.log_prob(observed_outputs) +
         math.log(args.expected_num_objects))

    b = (spurious_loc_dist.log_prob(observed_positions) +
         spurious_output_dist.log_prob(observed_outputs) +
         math.log(args.max_detections_per_frame - args.expected_num_objects))

    return torch.cat((a, b), dim=-1)


class Detector(nn.Module):
    # returns confidence of sensor sensing the object
    def __init__(self):
        super(Detector, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.linear.weight, 1.)
        nn.init.constant_(self.linear.bias, 0.)

    def __str__(self):
        return "Detector: w={}, b={}".format(self.linear.weight.item(), self.linear.bias.item())

    def forward(self, sensor_outputs):
        # x * w + b
        return torch.sigmoid(self.linear((sensor_outputs - 0.5).unsqueeze(-1)).squeeze(-1))


class DetectorTracker(nn.Module):
    def __init__(self, args):
        super(DetectorTracker, self).__init__()
        self.detector = Detector()
        self.num_objects = args.max_num_objects

    @poutine.broadcast
    def model(self, sensor_positions, sensor_outputs, args):
        bin_width = (args.x_max - args.x_min) / args.num_sensors
        noise_power = 10 ** (-args.PNR / 10)
        pyro.module("detectorTracker", self)
        with pyro.iarange("objects", self.num_objects):
            exists = pyro.sample("exists",
                                 dist.Bernoulli(args.expected_num_objects / self.num_objects))
            with poutine.scale(scale=exists):
                states = pyro.sample("states", dist.Normal(0., 1.).expand([2]).independent(1))
                positions = get_positions(states, args.num_frames)

        with pyro.iarange("detections", args.max_detections_per_frame):
            with pyro.iarange("time", args.num_frames):
                # The combinatorial part of the log prob is approximated to allow independence.
                assign = pyro.sample("assign",
                                     dist.Categorical(torch.ones(self.num_objects + 1)))
                assert assign.shape == (self.num_objects + 1, 1, args.num_frames, args.max_detections_per_frame)
                is_real = assign < self.num_objects
                with poutine.scale(scale=is_real.float()):
                    bogus_position = positions.new_zeros(args.num_frames, 1)
                    augmented_positions = torch.cat([positions, bogus_position], -1)
                    assert augmented_positions.shape == (args.num_frames, self.num_objects + 1), \
                        "augmented_positions.shape: {}".format(augmented_positions.shape)
                    predicted_positions = augmented_positions[:, assign]
                    assert predicted_positions.shape == (self.num_objects + 1, 1,
                                                         args.num_frames,
                                                         args.max_detections_per_frame), \
                        "predicted_positions.shape: {}".format(predicted_positions.shape)

                    sampled_positions = pyro.sample("real_observations",
                                                    dist.Normal(predicted_positions, bin_width))
                with poutine.scale(scale=(1 - is_real.float())):
                    spurious_positions = pyro.sample("spurious_observations",
                                                     dist.Uniform(args.x_min, args.x_max).expand(is_real.shape))
                assert sampled_positions.shape == is_real.shape,\
                    "sampled_positions.shape: {}, is_real.shape: {}".format(sampled_positions.shape, is_real.shape)
                assert spurious_positions.shape == is_real.shape, \
                    "spurious_positions.shape: {}, is_real.shape: {}".format(spurious_positions.shape, is_real.shape)

                sampled_positions[~is_real] = spurious_positions[~is_real]

                observation_hat = torch.stack([sampled_positions,
                                               is_real,  # confidence
                                               is_real,  # sensor output
                                               ], -1)
                sensor_hat = vector2raster(observation_hat, args)
                assert sensor_hat.shape == (args.num_frames, args.max_detections_per_frame), \
                    "sensor_hat.shape: {}".format(sensor_hat.shape)
                pyro.sample("sensor_output",
                            dist.Normal(sensor_hat, noise_power).independent(2),
                            obs=sensor_outputs)

    @poutine.broadcast
    def guide(self, sensor_positions, sensor_outputs, args):
        pyro.module('detectorTracker', self)
        bin_width = (args.x_max - args.x_min) / args.num_sensors
        confidence = self.detector.forward(sensor_outputs)
        observations = raster2vector(sensor_positions, sensor_outputs, confidence, args)
        states_loc = torch.randn(self.num_objects, 2, requires_grad=True)
        for em_iter in range(args.em_iters):
            states_loc = states_loc.detach()
            states_loc.requires_grad = True
            positions = get_positions(states_loc, args.num_frames)
            replicates = max(1, states_loc.shape[0] / args.expected_num_objects)
            # E-step: compute soft assignments
            with torch.no_grad():
                assign_logits = compute_assign_logits(positions, observations, replicates, args)
                exists_logits = compute_exists_logits(states_loc, replicates)
                assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                                          args.bp_iters, bp_momentum=args.bp_momentum)
                p_exists = assignment.exists_dist.probs
                p_assign = assignment.assign_dist.probs

            log_likelihood = detection_log_likelihood(positions, observations, args)
            loss = -(log_likelihood * p_assign).sum()
            # M-step
            states_loc, _ = newton_step(loss, states_loc, bin_width)

            if args.prune_threshold > 0.0:
                states_loc = states_loc[p_exists > args.prune_threshold]
                self.num_objects = states_loc.shape[0]
            if args.merge_radius >= 0.0:
                states_loc, _ = merge_points(states_loc, args.merge_radius)
                self.num_objects = states_loc.shape[0]
            warn_if_nan(states_loc, 'states_loc')

        positions = get_positions(states_loc, args.num_frames)
        replicates = max(1, states_loc.shape[0] / args.expected_num_objects)
        assign_logits = compute_assign_logits(positions, observations, replicates, args)
        exists_logits = compute_exists_logits(states_loc, replicates)
        assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                                  args.bp_iters, bp_momentum=args.bp_momentum)

        with pyro.iarange("objects", states_loc.shape[0]):
            exists = pyro.sample("exists", assignment.exists_dist, infer={"enumerate": "parallel"})
            with poutine.scale(scale=exists):
                pyro.sample("states", dist.Delta(states_loc).independent(1))
        with pyro.iarange("detections", observations.shape[1]):
            with pyro.iarange("time", args.num_frames):
                pyro.sample("assign", assignment.assign_dist, infer={"enumerate": "parallel"})
        return assignment, states_loc, observations


def main(args):
    viz = plot_utils.init_visdom(args.visdom)
    pyro.set_rng_seed(0)
    true_states, true_positions, sensor_positions, sensor_outputs, true_confidence = generate_sensor_data(args)
    dt = DetectorTracker(args)

    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    assignment, states_loc, observations = dt.guide(sensor_positions, sensor_outputs, args)
    p_exists = assignment.exists_dist.probs
    positions = get_positions(states_loc, args.num_frames)
    if viz is not None:
        plot_solution(sensor_outputs, observations, p_exists, positions,
                      true_positions, args, 'Before training', viz=viz)
        plot_utils.plot_exists_prob(p_exists, viz=viz)

    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    infer = SVI(dt.model, dt.guide, ClippedAdam({"lr": 0.1}), TraceEnum_ELBO(max_iarange_nesting=2))
    losses = []
    for epoch in range(args.svi_iters if not smoke_test else 2):
        loss = infer.step(sensor_positions, sensor_outputs, args)
        if epoch % 10 == 0:
            print("epoch {: >4d} loss = {}".format(epoch, loss))
        losses.append(loss)
    if viz is not None:
        viz.plot(losses)
    else:
        pyplot.figure().patch.set_color('white')
        pyplot.plot(losses)

    dt = DetectorTracker(args)
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    assignment, states_loc, observations = dt.guide(sensor_positions, sensor_outputs, args)
    p_exists = assignment.exists_dist.probs
    positions = get_positions(states_loc, args.num_frames)
    if viz is not None:
        plot_solution(sensor_outputs, observations, p_exists, positions,
                      true_positions, args, 'Before training', viz=viz)
        plot_utils.plot_exists_prob(p_exists, viz)


def parse_args(*args):
    from shlex import split
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-frames', default=40, type=int, help='number of frames')
    parser.add_argument('--max-detections-per-frame', default=50, type=int, help='max number of detections per frame')
    parser.add_argument('--max-num-objects', default=20, type=int, help='maximum number of objects')
    parser.add_argument('--expected-num-objects', default=2.0, type=float, help='expected number of objects')
    parser.add_argument('--num-sensors', default=100, type=int, help='number of sensors')
    parser.add_argument('--x-min', default=-2.5, type=float, help='x-min')
    parser.add_argument('--x-max', default=2.5, type=float, help='x-max')
    parser.add_argument('--PNR', default=5.0, type=float, help='power to noise ratio (in dB)')

    parser.add_argument('--bp-iters', default=50, type=int, help='number of BP iterations')
    parser.add_argument('--bp-momentum', default=0.5, type=float, help='BP momentum')
    parser.add_argument('--svi-iters', default=20, type=int, help='number of SVI iterations')
    parser.add_argument('--em-iters', default=10, type=int, help='number of EM iterations')
    parser.add_argument('--merge-radius', default=0.5, type=float, help='merge radius')
    parser.add_argument('--prune-threshold', default=1e-2, type=float, help='prune threshold')
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
    true_states, true_positions, sensor_positions, sensor_outputs, true_confidence = generate_sensor_data(args)
    true_num_objects = len(true_states)
    assert true_states.shape == (true_num_objects, 2), \
        "true_states.shape: {}".format(true_states.shape)
    assert true_positions.shape == (args.num_frames, true_num_objects), \
        "true_positions.shape: {}".format(true_positions.shape)


@pytest.mark.parametrize("inp", [torch.rand(5, 5), torch.rand(5, 5, 5)])
def test_detector(inp):
    d = Detector()
    out = d.forward(inp)
    assert out.shape == inp.shape, "input: {}, output:{}".format(inp.shape, out.shape)


@pytest.mark.parametrize("args", ['--no-visdom'])
def test_guide(args):
    if isinstance(args, str):
        args = parse_args(args)
    pyro.set_rng_seed(0)
    true_states, true_positions, sensor_positions, sensor_outputs, true_confidence = generate_sensor_data(args)

    dt = DetectorTracker(args)
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    assignment, states_loc, observations = dt.guide(sensor_positions, sensor_outputs, args)
    p_exists = assignment.exists_dist.probs
    positions = get_positions(states_loc, args.num_frames)
    assert p_exists.dim() == 1
    assert positions.shape[0] == args.num_frames


@pytest.mark.xfail(reason='Nothing works mr. duck :(')
@pytest.mark.parametrize("args", ['--no-visdom --svi-iters 2'])
def test_svi(args):
    if isinstance(args, str):
        args = parse_args(args)
    pyro.set_rng_seed(0)
    true_states, true_positions, sensor_positions, sensor_outputs, true_confidence = generate_sensor_data(args)

    dt = DetectorTracker(args)
    pyro.set_rng_seed(1)  # Use a different seed from data generation
    pyro.clear_param_store()
    infer = SVI(dt.model, dt.guide, ClippedAdam({"lr": 0.1}), TraceEnum_ELBO(max_iarange_nesting=2))
    losses = []
    for epoch in range(args.svi_iters if not smoke_test else 2):
        loss = infer.step(sensor_positions, sensor_outputs, args)
        if epoch % 10 == 0:
            print("epoch {: >4d} loss = {}".format(epoch, loss))
        losses.append(loss)


if __name__ == '__main__':
    args = args = parse_args()
    assert args.max_num_objects >= args.expected_num_objects
    assert args.x_max > args.x_min
    assert args.max_detections_per_frame >= args.max_num_objects
    main(args)
