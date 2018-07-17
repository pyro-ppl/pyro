import torch
import math
import pyro.distributions as dist


def get_positions(states_loc, num_frames):
    """
    Let's consider a model with deterministic dynamics, say sinusoids with known period but unknown phase and amplitude.
    """
    time = torch.arange(num_frames, dtype=torch.float) * 2 * math.pi / num_frames
    return torch.stack([time.cos(), time.sin()], -1).mm(states_loc.t())


def generate_data(args):
    """
    Generates data according to dynamics implemented by get_positions.
    """
    num_objects = int(round(args.expected_num_objects))  # Deterministic.
    states = dist.Normal(0., 1.).sample((num_objects, 2))

    # Detection model.
    emitted = dist.Bernoulli(args.emission_prob).sample((args.num_frames, num_objects))
    num_spurious = dist.Poisson(args.expected_num_spurious).sample((args.num_frames,))
    max_num_detections = int((num_spurious + emitted.sum(-1)).max())
    observations = torch.zeros(args.num_frames, max_num_detections, 2)  # position+confidence
    positions = get_positions(states, args.num_frames)
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
            observations[t, j:j + n, 0] = dist.Normal(0., 1.).sample((n,))
            observations[t, j:j + n, 1] = 1

    return states, positions, observations
