import pytest
from tracking_1d_multi import track_1d_objects
from datagen_utils import generate_observations, get_positions
from tracking_1d_multi import parse_args

import pyro
import os


def calculate_motmetrics(true_positions, inferred_positions, exp_dir=None):
    assert true_positions.shape[0] == inferred_positions.shape[0]
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(true_positions.shape[1]):
        C = (true_positions[f].unsqueeze(-1) - inferred_positions[f].unsqueeze(0)).pow(2).detach().tolist()
        acc.update(
            list(range(true_positions.shape[1])),  # Ground truth objects in this frame
            list(range(inferred_positions.shape[1])),  # Detector hypotheses in this frame
            C)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    if exp_dir is not None:
        summary.to_csv(os.path.join(exp_dir, "results.csv"), header=True, sep=',', mode='w')
    return summary


def xfail_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


@pytest.mark.parametrize("arg_string", [
    "--num-frames=20 --max-num-objects=2 --expected-num-objects=2 --emission-noise-scale=0.1 --seed=2",
    xfail_param("--num-frames=20 --max-num-objects=2 --expected-num-objects=2 --emission-noise-scale=0.1 --seed=1",
                reason="objects are too cluttered"),
    "--num-frames=20 --max-num-objects=40 --expected-num-objects=2 --emission-noise-scale=0.1 --seed=2",
    "--num-frames=20 --max-num-objects=40 --expected-num-objects=10 --emission-noise-scale=0.1 --seed=2",
    xfail_param("--num-frames=20 --max-num-objects=40 --expected-num-objects=10 --emission-noise-scale=0.1 --seed=1",
                reason="objects are too cluttered"),
    xfail_param("--num-frames=20 --max-num-objects=400 --expected-num-objects=10 --emission-noise-scale=0.1 --seed=1",
                reason="converges to single object"),
    "--num-frames=20 --max-num-objects=40 --expected-num-objects=5 --emission-noise-scale=0.2 --seed=2",
    "--num-frames=20 --max-num-objects=40 --expected-num-objects=5 --emission-noise-scale=0.1 --emission-prob=0.8 "
    "--seed=2",
    "--num-frames=20 --max-num-objects=40 --expected-num-objects=5 --emission-noise-scale=0.1 "
    "--expected-num-spurious=0.2 --seed=2",
    xfail_param("--num-frames=20 --max-num-objects=40 --expected-num-objects=5 --emission-noise-scale=0.1 "
                "--expected-num-spurious=0.2 --seed=1", reason="converges to single object+some residuals"),
    "--num-frames=20 --max-num-objects=40 --expected-num-objects=2 --emission-noise-scale=0.1 "
    "--expected-num-spurious=0.2 --seed=1", ])
@pytest.mark.filterwarnings('ignore::ImportWarning')  # pandas raise warning for some reason...
def test_1d_track_objects(arg_string):
    args = parse_args(arg_string + " --good-init -q --no-visdom")
    # generate data
    pyro.set_rng_seed(args.seed)
    true_states, true_positions, observations = generate_observations(args)
    true_num_objects = len(true_states)
    max_num_detections = observations.shape[1]
    assert true_states.shape == (true_num_objects, 2)
    assert true_positions.shape == (args.num_frames, true_num_objects)
    assert observations.shape == (args.num_frames, max_num_detections, 2)
    track_1d_objects(args, observations, true_states)

    inferred_states = pyro.param("states_loc")
    inferred_positions = get_positions(inferred_states, args.num_frames)
    inferred_ens = pyro.param("emission_noise_scale")
    old_metric = False
    if old_metric:
        true_states = true_states.unsqueeze(1)
        inferred_states = inferred_states.unsqueeze(0)
        dist_matrix = (true_states - inferred_states).pow(2).sum(-1).sqrt()
        dist_threshold = 0.2
        match = (dist_matrix <= dist_threshold).float()
        assert (match.sum(dim=1) > 0).all().item(), dist_matrix  # check all true objects have at least 1 matching
    else:
        full_exp_dir = os.path.join(args.exp_dir, args.exp_name) if args.exp_name is not None else None
        metrics = calculate_motmetrics(true_positions, inferred_positions, exp_dir=full_exp_dir)  # metrics is DataFrame
        # mota = metrics['mota']['acc']
        motp = metrics['motp']['acc']
        assert motp < 1e-3

    ens_threshold = 0.2
    assert ((inferred_ens - args.emission_noise_scale).abs() <= ens_threshold).item()
