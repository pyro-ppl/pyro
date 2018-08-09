from __future__ import absolute_import, division, print_function
import os
import pytest

import torch
import pyro

from datagen_utils import generate_observations, get_positions
from plot_utils import plot_solution, plot_exists_prob, init_plot_utils, plot_list
from experiment_utils import args2json
from tracking_1d_multi import track_1d_objects, parse_args, guide
from test_tracking_1d_multi import calculate_motmetrics


def pytest_generate_tests(metafunc):
    args_list = generate_list_of_experiments()
    if 'args' in metafunc.fixturenames:
        metafunc.parametrize("args", args_list)


def generate_list_of_experiments():
    from itertools import product
    num_frames_list = [2, 10, 20]
    max_num_objects_list = [2, 10, 80, 400]
    expected_num_objects_list = [2, 5, 10, 20]
    expected_num_spurious_list = [0.00001, 0.1, 1, 2, 5, 10]
    emission_prob_list = [.7, .8, .9, .9999]
    emission_noise_scale_list = [0.05, 0.1, 0.2, 0.3]
    result = product(num_frames_list, max_num_objects_list, expected_num_objects_list,
                     expected_num_spurious_list, emission_prob_list, emission_noise_scale_list
                     )
    arg_strings = []
    for each in result:
        (num_frames, max_num_objects, expected_num_objects,
         expected_num_spurious, emission_prob, emission_noise_scale) = each
        if max_num_objects < expected_num_objects:
            continue
        arg_string = " ".join(
            ["--exp-name {}".format('exp' + str(len(arg_strings))),
             "--num-frames={}".format(num_frames),
             "--max-num-objects={}".format(max_num_objects),
             "--expected-num-objects={}".format(expected_num_objects),
             "--expected-num-spurious={}".format(expected_num_spurious),
             "--emission-prob={}".format(emission_prob),
             "--emission-noise-scale={}".format(emission_noise_scale),
             "--merge-radius=1e-3",
             "--prune-threshold=-1e-2",
             "--good-init",
             "--no-visdom",
             "--seed=2"
             ])
        arg_strings.append(arg_string)
    return arg_strings


def test_experiment(args):
    args = parse_args(args + " --good-init -q --no-visdom")
    # generate data
    pyro.set_rng_seed(args.seed)
    true_states, true_positions, observations = generate_observations(args)
    true_num_objects = len(true_states)
    max_num_detections = observations.shape[1]
    assert true_states.shape == (true_num_objects, 2)
    assert true_positions.shape == (args.num_frames, true_num_objects)
    assert observations.shape == (args.num_frames, max_num_detections, 2)
    if not args.quiet:
        print("generated {:d} detections from {:d} objects".format((observations[..., -1] > 0).long().sum(),
                                                                   true_num_objects))
        print('true_states = {}'.format(true_states))

    # initialization
    viz, full_exp_dir = init_plot_utils(args)
    if full_exp_dir is not None:
        args2json(args, os.path.join(full_exp_dir, 'config.json'))
    env = str(args)

    losses, ens = track_1d_objects(args, observations, true_states)

    # run visualizations
    if (viz is not None) or (full_exp_dir is not None):
        plot_list(losses, "Loss", viz=viz, env=env, fig_dir=full_exp_dir)
        plot_list(ens, "Emission Noise Scale", viz=viz, env=env, fig_dir=full_exp_dir)

        # Run guide once and plot final result
        with torch.no_grad():
            states_loc = pyro.param("states_loc")
            positions = get_positions(states_loc, args.num_frames)
            p_exists = guide(args, observations)
        plot_solution(observations, p_exists, positions, true_positions, args,
                      pyro.param("emission_noise_scale").item(),
                      'After inference', viz=viz, env=env, fig_dir=full_exp_dir)
        plot_exists_prob(p_exists, viz, env=env, fig_dir=full_exp_dir)

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


if __name__ == '__main__':
    pytest.main(["-n=auto", "run_experiments.py::test_experiment"])
