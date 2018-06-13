from __future__ import absolute_import, division, print_function

import argparse
import math

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam

# TODO(alicanb) implement VonMises.sample() and switch to VonMises throughout this file.
VON_MISES_HAS_SAMPLE = False


def wrap_position(state):
    """Wraps to the interval ``[0, 1)``."""
    return (state + state.floor()).fmod(1)


def wrap_displacement(state):
    """Wraps to the interval ``[-0.5, 0.5)``."""
    return wrap_position(state + 0.5) - 0.5


class PartialMatching(dist.TorchDistribution):
    def __init__(self, exists, assign_logits):
        raise NotImplementedError


# This models multiple objects randomly walking around the unit 2-torus,
# as is common in video games with wrapped edges.
@poutine.broadcast
def state_model(args):
    with pyro.iarange('objects', args.max_num_objects):
        # This is equivalent to sampling num_objects from
        # Binomial(max_num_objects, expected_num_objects / max_num_objects).
        exists = pyro.sample('exists', dist.Bernoulli(args.expected_num_objects / args.max_num_objects))
        with poutine.scale(None, exists):
            # Initialize uniformly in unit square.
            states = torch.empty(args.num_frames, args.max_num_objects, 2)
            states[0] = pyro.sample('state_0', dist.Uniform(0., 1.).expand([2]))

            # Randomly walk in time.
            for t in range(1, args.num_frames):
                if VON_MISES_HAS_SAMPLE:
                    states[t] = pyro.sample('state_{}'.format(t),
                                            dist.VonMises(states[t - 1], args.transition_noise_scale).expand([2]))
                else:
                    transition_noise = pyro.sample('transition_noise_{}'.format(t),
                                                   dist.Normal(0., args.transition_noise_scale).expand([2]))
                    states[t] = wrap_position(states[t - 1] + transition_noise)
    return exists, states


@poutine.broadcast
def assignment_model(args, exists, states, observations):
    max_num_detections = max(map(len, observations))
    assign_logits = torch.empty(args.num_frames, max_num_detections, args.max_num_objects + 1)
    assign_logits[..., :-1] = -float('inf')
    assign_logits[..., -1] = 0.
    # TODO populate assign_logits using args.emission_prob and args.expected_num_false_alarms
    raise NotImplementedError
    if False:  # TODO(fritzo) implement PartialMatching distribution
        return pyro.sample("assign", dist.PartialMatching(exists, assign_logits))
    else:
        return pyro.sample("assign", dist.Categorical(assign_logits))


@poutine.broadcast
def detection_model(args, exists, states, assign, observations=None):
    augmented_states = torch.nn.functional.pad(states, (0, 1), "constant", 0.0)  # FIXME
    is_real = (assign != args.max_num_objects)

    # This requires two branches due to the operation required to combine real+spurious.
    if observations is None:
        with pyro.scale(None, is_real.float()):
            if VON_MISES_HAS_SAMPLE:
                real_observations = pyro.sample('real_observations',
                                                dist.VonMises(augmented_states[assign], args.emission_noise_scale))
            else:
                emission_noise = pyro.sample('emission_noise',
                                             dist.Normal(0., args.emission_noise_scale))
                real_observations = wrap_position(states, emission_noise)
        with pyro.scale(None, (~is_real).float()):
            spurious_observations = pyro.sample('spurious_observations', dist.Uniform(0., 1.).expand([2]))
        observations = torch.empty(real_observations.shape)
        observations[is_real] = real_observations[is_real]
        observations[~is_real] = spurious_observations[~is_real]
    else:
        with pyro.scale(None, is_real.float()):
            if VON_MISES_HAS_SAMPLE:
                real_observations = pyro.sample('real_observations',
                                                dist.VonMises(augmented_states[assign], args.emission_noise_scale),
                                                obs=observations)
            else:
                emission_noise = pyro.sample('emission_noise',
                                             dist.Normal(0., args.emission_noise_scale),
                                             obs=wrap_displacement(states - observations))
        with pyro.scale(None, (~is_real).float()):
            pyro.sample('spurious_observations', dist.Uniform(0., 1.).expand([2]),
                        obs=observations)

    return observations


def model(args, observations=None):
    exists, states = state_model(args)
    assign = assignment_model(args, exists, states, observations)
    observations = detection_model(args, exists, states, assign, observations)
    return exists, states, assign, observations


# This guide uses a smart assignment solver but a naive state estimator.
# A smarter implementation would use message passing also for state estimation,
# e.g. a Kalman filter-smoother.
@poutine.broadcast
def guide(args, observations):
    # Initialize from the prior.
    states = pyro.param('states_param', lambda: poutine.block(state_model)(args))
    states = wrap_position(states)  # Account for drift during optimization.

    # Solve soft assignment problem.
    # TODO(eb8680,fritzo) replace this hand computation with poutines.
    exists_logits = torch.empty(args.max_num_objects).fill_(math.log(args.max_num_objects / args.expected_num_objects))
    emission_noise = wrap_displacement(states.unsqueeze(-1) - observations)
    assign_logits = (dist.Normal(0., args.emission_noise_scale).expand([2]).log_prob(emission_noise) -
                     dist.Uniform(0., 0.5).expand([2]).log_prob(emission_noise))
    assignment = MarginalAssignmentPersistent(exists_logits, assign_logits)
    # Hereafter we make the mean-field approximation that object existence is approximately
    # independent of object-detection assignment.

    with pyro.iarange('objects', args.max_num_objects):
        exists = pyro.sample('exists', assignment.exists_dist,
                             infer={'enumerate': 'parallel'})
        with poutine.scale(None, exists):
            for t in range(1, args.num_frames):
                pyro.sample('states', dist.Delta(states, event_dim=2))

    with pyro.iarange('time', observations.shape[0], dim=-2):
        with pyro.iarange('detections', observations.shape[1], dim=-1):
            pyro.sample('assign', assignment.assignment_dist)


def main(args):
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)

    # Generate data.
    true_states, observations = model(args)
    num_objects = true_states.shape[1]
    assert true_states.shape == (args.num_frames, num_objects, 2)
    print('generated {} objects'.format(num_objects))

    # Train.
    infer = SVI(model, guide, Adam({'lr': 0.02}), TraceEnum_ELBO(max_iarange_nesting=2))
    for step in range(args.num_epochs):
        loss = infer.step(args, observations)
        print('epoch {: >4d} loss = {}'.format(step, loss))

    # Evaluate.
    # TODO(null-a) compute MOTP, MOTA


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-epochs', default=100, type=int)
    parser.add_argument('--num-frames', default=5, type=int)
    parser.add_argument('--max-num-objects', default=4, type=int)
    parser.add_argument('--expected-num-objects', default=3.0, type=float)
    parser.add_argument('--expected-num-false-alarms', default=1.0, type=float)
    parser.add_argument('--emission-prob', default=0.8, type=float)
    parser.add_argument('--transition-noise-scale', default=0.1, type=float)
    parser.add_argument('--emission-noise-scale', default=0.1, type=float)
    args = parser.parse_args()
    main(args)
