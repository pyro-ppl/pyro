import torch
import math
import pyro.distributions as dist


def get_positions(states, num_frames):
    """
    Let's consider a model with deterministic dynamics, say sinusoids with known period but unknown phase and amplitude.
    """
    time = torch.arange(num_frames, dtype=torch.float) * 2 * math.pi / num_frames
    return torch.stack([time.cos(), time.sin()], -1).mm(states.t())


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
            observations[t, j:j + n, 0] = dist.Uniform(-3., 3.).sample((n,))
            observations[t, j:j + n, 1] = 1

    return states, positions, observations


def generate_sensor_data(args):
    num_objects = int(round(args.expected_num_objects))  # Deterministic.
    states = dist.Normal(0., 1.).sample((num_objects, 2))
    confidence = torch.empty(args.num_frames, args.num_sensors)
    positions = get_positions(states, args.num_frames)
    noise_power = 10 ** (-args.PNR / 10)
    noise_dist = dist.Normal(0, noise_power)
    # confidence is number of objects each sensor is sensing
    for t in range(args.num_frames):
        confidence[t] = torch.histc(positions[t], args.num_sensors, args.x_min, args.x_max)
    # if sensors are saturated: can't diff btw 1 object and multiple objects.
    confidence[confidence > 1] = 1
    # AWGN model
    sensor_outputs = noise_dist.sample(confidence.shape) + confidence
    bin_width = (args.x_max - args.x_min) / args.num_sensors
    sensor_positions = torch.arange(args.x_min, args.x_max, bin_width) + bin_width / 2
    return states, positions, sensor_positions, sensor_outputs, confidence


def vector2raster(vec, args):
    # this is essentially the inverse of histc
    def pos2sensoridx(pos):
        return torch.floor((pos - args.x_min) /
                           (args.x_max - args.x_min) *
                           args.num_sensors).long()
    pos = vec[..., 0]
    sensor_outputs = vec[..., 2]
    raster = torch.zeros(vec.shape[:-3] + (args.num_frames, args.num_sensors))
    raster.scatter_(-1, pos2sensoridx(pos), sensor_outputs)
    return raster


def raster2vector(sensor_positions, sensor_outputs, confidence, args):
    _, idx = torch.sort(confidence, dim=-1, descending=True)
    idx = idx[..., :args.max_detections_per_frame]
    pos = sensor_positions[idx]
    conf = torch.gather(confidence, -1, idx)
    out = torch.gather(sensor_outputs, -1, idx)
    return torch.stack((pos, conf, out), -1)


def test_vector2raster():
    args = type('Args', (object,), {})  # A fake ArgumentParser.parse_args() result.
    args.num_frames = 40
    args.max_detections_per_frame = 100
    args.max_num_objects = 90
    args.expected_num_objects = 2.
    args.PNR = 10
    args.num_sensors = 100
    args.x_min, args.x_max = -2.5, 2.5

    _, _, sensor_positions, sensor_outputs, true_confidence = generate_sensor_data(args)
    obs = raster2vector(sensor_positions, sensor_outputs, sensor_outputs, args)
    obs2 = obs.unsqueeze(0).expand(2, -1, -1, -1)
    sensor_outputs2 = vector2raster(obs2, args)
    assert (sensor_outputs2[0] == sensor_outputs).all()


def test_raster2vector():
    args = type('Args', (object,), {})  # A fake ArgumentParser.parse_args() result.
    args.num_frames = 40
    args.max_detections_per_frame = 100
    args.max_num_objects = 90
    args.expected_num_objects = 2.
    args.PNR = 10
    args.num_sensors = 100
    args.x_min, args.x_max = -2.5, 2.5

    _, _, sensor_positions, sensor_outputs, true_confidence = generate_sensor_data(args)
    obs = raster2vector(sensor_positions, sensor_outputs, sensor_outputs, args)
    sensor_outputs = vector2raster(obs, args)
    obs2 = obs.unsqueeze(0).expand(2, -1, -1, -1)
    sensor_outputs2 = vector2raster(obs2, args)
    assert (sensor_outputs2[0] == sensor_outputs).all()
