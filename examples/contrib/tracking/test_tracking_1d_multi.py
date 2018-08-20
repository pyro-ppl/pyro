import pytest
from tracking_1d_multi import track_1d_objects
from datagen_utils import generate_observations, get_positions
from tracking_1d_multi import parse_args
import pyro
import os


def calculate_metrics(true_positions, inferred_positions, true_ens, inferred_ens, exp_dir=None):
    assert true_positions.shape[0] == inferred_positions.shape[0]
    import motmetrics as mm
    import pandas

    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(true_positions.shape[0]):
        C = (true_positions[f].unsqueeze(-1) - inferred_positions[f].unsqueeze(0)).pow(2).detach().tolist()
        acc.update(
            list(range(true_positions.shape[1])),  # Ground truth objects in this frame
            list(range(inferred_positions.shape[1])),  # Detector hypotheses in this frame
            C)
    metrics = mm.metrics.create().compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')
    metrics.loc[:, 'ens_error'] = pandas.Series((inferred_ens - true_ens).abs().item(), index=metrics.index)
    if exp_dir is not None:
        metrics.to_csv(os.path.join(exp_dir, "results.csv"), header=True, sep=',', mode='w')
    return metrics


def xfail_param(*args, **kwargs):
    return pytest.param(*args, marks=[pytest.mark.xfail(**kwargs)])


@pytest.mark.parametrize("arg_string", [
    "--num-frames=20 --max-num-objects=2 --expected-num-objects=2 --emission-noise-scale=0.1 --seed=2",
    "--num-frames=20 --max-num-objects=2 --expected-num-objects=2 --emission-noise-scale=0.1 --seed=1",
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
    "--num-frames=20 --max-num-objects=40 --expected-num-objects=5 --emission-noise-scale=0.1 "
    "--expected-num-spurious=0.2 --seed=1",
    xfail_param("--num-frames=20 --max-num-objects=40 --expected-num-objects=2 --emission-noise-scale=0.1 "
                "--expected-num-spurious=0.2 --seed=2", reason="converges to single object"),
                 ])
@pytest.mark.filterwarnings('ignore::ImportWarning')  # pandas raise warning for some reason...
@pytest.mark.filterwarnings('ignore::DeprecationWarning')  # pandas raise warning for some reason...
def test_1d_track_objects(arg_string):
    args = parse_args(arg_string + " --good-init=both -q --no-visdom")
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
        metrics = calculate_metrics(true_positions, inferred_positions,
                                    args.emission_noise_scale, inferred_ens)  # metrics is pandas DataFrame
        # mota = metrics['mota']['acc']
        motp = metrics['motp']['acc']
        assert motp < 1e-2

    ens_threshold = 0.2
    assert ((inferred_ens - args.emission_noise_scale).abs() <= ens_threshold).item()
