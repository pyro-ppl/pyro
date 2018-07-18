import torch
import math
import pyro.distributions as dist


def get_positions(states_loc, num_frames):
    """
    Let's consider a model with deterministic dynamics, say sinusoids with known period but unknown phase and amplitude.
    """
    time = torch.arange(num_frames, dtype=torch.float) * 2 * math.pi / num_frames
    return torch.stack([time.cos(), time.sin()], -1).mm(states_loc.t())


def generate_observations(args):
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


def generate_sensor_data(args):
    num_objects = int(round(args.expected_num_objects))  # Deterministic.
    states = dist.Normal(0., 1.).sample((num_objects, 2))
    confidence = torch.empty(args.num_frames, args.num_sensors)
    positions = get_positions(states, args.num_frames)
    noise_power = 10 ** (-args.PNR / 10)
    noise_dist = dist.Normal(0, noise_power)
    # confidence is number of objects indicating sensor senses object/s
    for t in range(args.num_frames):
        confidence[t] = torch.histc(positions[t], args.num_sensors, args.x_min, args.x_max)
    # if sensors are saturated: can't diff btw 1 object and multiple objects.
    confidence[confidence > 1] = 1
    # AWGN model
    sensor_outputs = noise_dist.sample(confidence.shape) + confidence
    bin_width = (args.x_max - args.x_min) / args.num_sensors
    sensor_positions = torch.arange(args.x_min, args.x_max, bin_width) + bin_width / 2
    return states, positions, sensor_positions, sensor_outputs, confidence


def obs2sensor(obs, args):
    sensor_outputs = torch.zeros(args.num_frames, args.num_sensors)
    # this is essentially the inverse of histc
    pos2sensoridx = lambda pos: torch.floor((pos - args.x_min) /
                                            (args.x_max - args.x_min) *
                                            args.num_sensors).long()
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            if obs[i, j, 1] >= 0.0:
                sensor_outputs[i, pos2sensoridx(obs[i, j, 0])] = obs[i, j, 2]
    return sensor_outputs


def sensor2obs(sensor_positions, sensor_outputs, confidence, args):
    observations = torch.zeros(sensor_outputs.shape[-2], args.max_detections_per_frame, 3)
    for i in range(args.num_frames):
        k = 0
        _, idx = torch.sort(confidence[i], descending=True)
        for j in range(min(sensor_positions.shape[0], int(args.max_detections_per_frame))):
            observations[i, j, 0] = sensor_positions[idx[j]]
            observations[i, j, 1] = confidence[i, idx[j]]
            observations[i, j, 2] = sensor_outputs[i, idx[j]]
    return observations
