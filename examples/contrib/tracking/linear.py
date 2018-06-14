from __future__ import absolute_import, division, print_function

import argparse
import math

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam


def get_dynamics(num_frames):
    time = torch.arange(num_frames) / 10
    return torch.stack([time.cos(), time.sin()], -1)


def generate_data(args):
    # Object model.
    num_objects = int(round(args.expected_num_objects))  # Deterministic.
    states = dist.Normal(0., 1.).sample((num_objects, 2))

    # Detection model.
    emitted = dist.Bernoulli(args.emission_prob).sample((args.num_frames, num_objects))
    num_spurious = dist.Poisson(args.expected_num_spurious).sample((args.num_frames,))
    max_num_detections = int((num_spurious + emitted.sum(-1)).max())
    observations = torch.zeros(args.num_frames, max_num_detections, 1+1)  # position+confidence
    positions = get_dynamics(args.num_frames).mm(states.t())
    noisy_positions = dist.Normal(positions, args.emission_noise_scale).sample()
    for t in range(args.num_frames):
        j = 0
        for i, e in enumerate(emitted[t]):
            if e:
                observations[t, j, 0] = noisy_positions[t, i]
                observations[t, j, 1] = 1
                j += 1
        n = int(num_spurious[t])
        if n:
            observations[t, j:j+n, 0] = dist.Normal(0., 1.).sample((n,))
            observations[t, j:j+n, 1] = 1

    return states, positions, observations


@poutine.broadcast
def model(args, observations):
    with pyro.iarange("objects", args.max_num_objects):
        exists = pyro.sample("exists", dist.Bernoulli(args.expected_num_objects / args.max_num_objects))
        with poutine.scale(scale=exists):
            states = pyro.sample("states", dist.Normal(0., 1.).expand([2]).independent(1))
            positions = get_dynamics(args.num_frames).mm(states.t())
    with pyro.iarange("detections", observations.shape[1]):
        with pyro.iarange("time", args.num_frames):
            # The combinatorial part of the log prob is approximated to allow independence.
            assign = pyro.sample("assign", dist.Categorical(torch.ones(args.max_num_objects + 1)))
            is_observed = (observations[..., -1] > 0)
            is_spurious = (assign == args.max_num_objects)
            is_real = is_observed & ~is_spurious
            num_observed = is_observed.float().sum(-1, True)
            pyro.sample("is_real", dist.Bernoulli(args.expected_num_objects / num_observed),
                        obs=is_real.float())
            pyro.sample("is_spurious", dist.Bernoulli(args.expected_num_spurious / num_observed),
                        obs=is_spurious.float())

            # The remaining continuous part is exact.
            observed_positions = observations[..., 0]
            with poutine.scale(scale=is_real.float()):
                bogus_position = positions.new_zeros(args.num_frames, 1)
                augmented_positions = torch.cat([positions, bogus_position], -1)
                predicted_positions = augmented_positions[:, assign]
                pyro.sample("real_observations",
                            dist.Normal(predicted_positions, args.emission_noise_scale),
                            obs=observed_positions)
            with poutine.scale(scale=is_spurious.float()):
                pyro.sample("spurious_observations", dist.Normal(0., 1.),
                            obs=observed_positions)


# This guide uses a smart assignment solver but a naive state estimator.
# A smarter implementation would use message passing also for state estimation,
# e.g. a Kalman filter-smoother.
@poutine.broadcast
def guide(args, observations):
    # Initialize states randomly from the prior.
    states_loc = pyro.param("states_loc", lambda: torch.randn(args.max_num_objects, 2))
    states_scale = pyro.param("states_scale", lambda: torch.ones(states_loc.shape),
                              constraint=constraints.positive)
    positions = get_dynamics(args.num_frames).mm(states_loc.t())

    # Solve soft assignment problem.
    real_dist = dist.Normal(positions.unsqueeze(-2), args.emission_noise_scale)
    spurious_dist = dist.Normal(0., 1.)
    observed_positions = observations[..., 0].unsqueeze(-1)
    assign_logits = real_dist.log_prob(observed_positions) - spurious_dist.log_prob(observed_positions)
    exists_logits = torch.empty(args.max_num_objects).fill_(math.log(args.max_num_objects / args.expected_num_objects))
    assignment = MarginalAssignmentPersistent(exists_logits, assign_logits)

    with pyro.iarange("objects", args.max_num_objects):
        exists = pyro.sample("exists", assignment.exists_dist, infer={"enumerate": "parallel"})
        with poutine.scale(scale=exists):
            pyro.sample("states", dist.Normal(states_loc, states_scale).independent(1))
    with pyro.iarange("detections", observations.shape[1]):
        with pyro.iarange("time", args.num_frames):
            pyro.sample("assign", assignment.assign_dist, infer={"enumerate": "parallel"})


def main(args):
    assert args.max_num_objects >= args.expected_num_objects
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)

    # Generate data.
    true_states, true_positions, observations = generate_data(args)
    true_num_objects = len(true_states)
    max_num_detections = observations.shape[1]
    assert true_states.shape == (true_num_objects, 2)
    assert true_positions.shape == (args.num_frames, true_num_objects)
    assert observations.shape == (args.num_frames, max_num_detections, 1+1)
    print("generated {:d} detections from {:d} objects".format(
        (observations[..., -1] > 0).long().sum(), true_num_objects))

    # Train.
    infer = SVI(model, guide, Adam({"lr": 0.02}), TraceEnum_ELBO(max_iarange_nesting=2))
    for step in range(args.num_epochs):
        loss = infer.step(args, observations)
        print("epoch {: >4d} loss = {}".format(step, loss))

    # Evaluate.
    # TODO(alicanb) compute MOTP, MOTA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("--num-frames", default=5, type=int)
    parser.add_argument("--max-num-objects", default=4, type=int)
    parser.add_argument("--expected-num-objects", default=2.0, type=float)
    parser.add_argument("--expected-num-spurious", default=1.0, type=float)
    parser.add_argument("--emission-prob", default=0.8, type=float)
    parser.add_argument("--transition-noise-scale", default=0.1, type=float)
    parser.add_argument("--emission-noise-scale", default=0.1, type=float)
    args = parser.parse_args()
    main(args)
