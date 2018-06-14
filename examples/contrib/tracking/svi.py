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
# This involves fixing [0,1) vs [0,2 pi) mismaches and eliminating all uses of Normal
# and most uses of the wrap_*() functions.
VON_MISES_HAS_SAMPLE = False


def wrap_position(state):
    """Wraps to the interval ``[0, 1)``."""
    return (state + state.floor()).fmod(1)


def wrap_displacement(state):
    """Wraps to the interval ``[-0.5, 0.5)``."""
    return wrap_position(state + 0.5) - 0.5


# This models multiple objects randomly walking around the unit 2-torus,
# as is common in video games with wrapped edges.
@poutine.broadcast
def state_model(args):
    with pyro.iarange('objects', args.max_num_objects):
        # This is equivalent to sampling num_objects from
        # Binomial(max_num_objects, expected_num_objects / max_num_objects)
        # which approximates Poisson(expected_num_objects) for large max_num_objects.
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
def assignment_model(args, exists, observations=None):
    # This requires two branches due to the deterministic dependency.
    if observations is None:
        with pyro.iarange("objects", args.max_num_objects):
            with pyro.iarange("time_emitted", args.num_frames):
                emitted = pyro.sample("emitted", dist.Bernoulli(args.emission_prob * exists.float()))
        with pyro.iarange("time_spurious", args.num_frames):
            num_false_alarms = pyro.sample("num_false_alarms",
                                           dist.Poisson(args.expected_num_false_alarms))
        max_num_detections = int((num_false_alarms + emitted.sum(-1)).max())
        assign = torch.empty(args.num_frames, max_num_detections, dtype=torch.long)
        assign.fill_(-1)  # -1 denotes no observation
        # TODO(fritzo) vectorize this.
        for t in range(args.num_frames):
            j = 0
            for i, e in enumerate(emitted[t]):
                if e:
                    assign[t, j] = i
                    j += 1
            assign[t, j:j+num_false_alarms[t]] = args.max_num_objects
        return assign
    else:
        with pyro.iarange("detections", args.max_num_objects):
            with pyro.iarange("time_assign", args.num_frames):
                is_observed = (observations[..., -1] > 0)
                with poutine.scale(scale=is_observed.float()):
                    # This ignores dependency on exists, but should only be off by a constant factor.
                    assign = pyro.sample("assign", dist.Categorical(torch.ones(args.max_num_objects + 1)))
                    assert assign.shape == (args.max_num_objects + 1, 1, args.num_frames, max_num_detections)
                assign[~is_observed] = -1
        with pyro.iarange("objects", args.max_num_objects):
            with pyro.iarange("time_emitted", args.num_frames):
                emitted = torch.zeros(assign.shape[:-1] + (args.max_num_objects + 1,))
                emitted[assign] = 1
                emitted = emitted[..., :-1]
                # TODO(fritzo,alicanb) Remove dependence on both of (assign,exists).
                pyro.sample("emitted", dist.Bernoulli(args.emission_prob * exists.float()),
                            obs=emitted)
        with pyro.iarange("time_spurious", args.num_frames):
            num_false_alarms = (assign == args.max_num_objects).long().sum(-1)
            pyro.sample("num_false_alarms", dist.Poisson(args.expected_num_false_alarms),
                        obs=num_false_alarms)


@poutine.broadcast
def detection_model(args, exists, states, assign, observations=None):
    # Append a bogus state for false detections.
    bogus_state = states.new_zeros(args.num_frames, 1, 2)
    augmented_states = torch.cat([states, bogus_state], 1)
    is_observed = (assign != -1)
    is_spurious = (assign == args.max_num_objects)
    is_real = is_observed & ~is_spurious

    # This requires two branches due to the deterministic dependency.
    if observations is None:
        with pyro.scale(None, is_real.float()):
            if VON_MISES_HAS_SAMPLE:
                observed_state = pyro.sample('real_observations',
                                             dist.VonMises(augmented_states[assign], args.emission_noise_scale))
            else:
                emission_noise = pyro.sample('emission_noise',
                                             dist.Normal(0., args.emission_noise_scale))
                observed_state = wrap_position(states, emission_noise)
        with pyro.scale(None, is_spurious.float()):
            observed_state[is_spurious] = pyro.sample(
                'spurious_observations', dist.Uniform(0., 1.).expand([2]))[is_spurious]
        confidence = is_observed.float() * args.emission_prob
        observations = torch.cat([observed_state, confidence.unsqueeze(-1)], -1)
    else:
        observed_states = observations[..., :-1]
        with pyro.scale(None, is_real.float()):
            if VON_MISES_HAS_SAMPLE:
                pyro.sample('real_observations',
                            dist.VonMises(augmented_states[assign], args.emission_noise_scale),
                            obs=observed_states)
            else:
                pyro.sample('emission_noise',
                            dist.Normal(0., args.emission_noise_scale),
                            obs=wrap_displacement(states - observed_states))
        with pyro.scale(None, is_spurious.float()):
            pyro.sample('spurious_observations', dist.Uniform(0., 1.).expand([2]),
                        obs=observed_states)

    max_num_detections = assign.shape[1]
    assert observations.shape == (args.num_frames, max_num_detections, 2 + 1)
    return observations


def model(args, observations=None):
    exists, states = state_model(args)
    assign = assignment_model(args, exists, observations)
    observations = detection_model(args, exists, states, assign, observations)
    return exists, states, assign, observations


# This guide uses a smart assignment solver but a naive state estimator.
# A smarter implementation would use message passing also for state estimation,
# e.g. a Kalman filter-smoother.
@poutine.broadcast
def guide(args, observations):
    # Initialize states randomly from the prior.
    states_loc = pyro.param('states_loc', lambda: poutine.block(state_model)(args)[1])
    states_scale = pyro.param('states_scale', torch.ones(states_loc.shape))
    states = pyro.sample('states', dist.Normal(states_loc, states_scale))
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
        exists = pyro.sample('exists', assignment.exists_dist, infer={'enumerate': 'parallel'})
        with poutine.scale(None, exists):
            for t in range(1, args.num_frames):
                pyro.sample('states', dist.Delta(states, event_dim=2))

    with pyro.iarange('time', args.num_frames, dim=-2):
        with pyro.iarange('detections', observations.shape[1], dim=-1):
            pyro.sample('assign', assignment.assignment_dist, infer={'enumerate': 'parallel'})


def main(args):
    assert args.max_num_objects >= args.expected_num_objects
    pyro.enable_validation(True)
    pyro.set_rng_seed(0)

    # Generate data.
    true_exists, true_states, true_assign, observations = model(args)
    max_num_detections = observations.shape[1]
    assert true_exists.shape == (args.max_num_objects,)
    assert true_states.shape == (args.num_frames, args.max_num_objects, 2)  # TODO Should existence change in time?
    assert true_assign.shape == (args.num_frames, max_num_detections)
    assert observations.shape == (args.num_frames, max_num_detections, 2 + 1)
    print('generated {:d} detections from {:d} objects'.format(
        (observations[..., -1] > 0).long().sum(), true_exists.long().sum()))

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
