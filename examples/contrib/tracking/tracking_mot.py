from __future__ import absolute_import, division, print_function
import math
import argparse
import os
import sys
import torch

import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.tracking.assignment import MarginalAssignmentPersistent
from pyro.contrib.tracking.hashing import merge_points
from pyro.ops.newton import newton_step
from pyro.util import warn_if_nan

from mot_utils import read_mot, write_mot

"""
MOT example. Requires pymotutils for visualization. You can install it by `pip install -r requirements.txt`
"""


def download_mot17(mot17_dir):
    """Download the MOT17 data."""
    from six.moves import urllib
    import zipfile

    def reporthook(count, block_size, total_size):
        progress_size = int(count * block_size)
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\rDownloading... {}%, {} MB".format(percent, progress_size // (1024 * 1024)))
        sys.stdout.flush()
    # download files
    try:
        os.makedirs(mot17_dir)
    except OSError as e:
        import errno
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    url = "https://motchallenge.net/data/MOT17.zip"
    filename = url.rpartition('/')[2]
    file_path = os.path.join(mot17_dir, filename)
    urllib.request.urlretrieve(url, file_path, reporthook=reporthook)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(mot17_dir)
    os.remove(file_path)
    print("done!")


def init_from_gt(gt_filename, args):
    frames, object_id, positions, sizes, features = read_mot(gt_filename, zero_based=True)
    assert (object_id != -1).all(), 'Some objects have id -1 in gt file.'
    num_objects = object_id.max().item() + 1
    pos = torch.rand(args.max_num_objects, 3)
    for i in range(num_objects):
        pos[i, :2] = positions[object_id == i].mean(dim=0)
        pos[i, 2] = sizes[object_id == i].mean()
    return pos


def init_from_first_frame(frames, positions, sizes):
    idx = frames == 0
    pos = torch.cat((positions[idx], sizes[idx].mean(-1).unsqueeze(-1)), -1)
    pos = torch.cat((pos, torch.rand(args.max_num_objects -
                                     pos.shape[0], pos.shape[1])), 0)
    return pos


def compute_exists_logits(positions, observations, args):
    return positions.new_empty(positions.shape[0]).fill_(-math.log(positions.shape[0]))


def compute_assign_logits(positions, observations, args):
    assert positions.dim() == 2 and positions.shape[1] == 3
    log_likelihood = detection_log_likelihood(positions, observations, args)
    assign_logits = log_likelihood[..., :-1] - log_likelihood[..., -1:]
    assign_logits[log_likelihood[..., 0] == -float('inf')] = -float('inf')
    return assign_logits


def detection_log_likelihood(positions, observations, args):
    assert positions.dim() == 2 and positions.shape[1] == 3
    real_dist = dist.Normal(positions, args.position_noise_scale).independent(1)
    spurious_dist = dist.Uniform(torch.zeros(2),
                                 torch.tensor([args.ymax, args.xmax]))
    spurious_dist = spurious_dist.independent(1)
    is_observed = (observations[..., -1] > 0)
    observed_positions = observations[..., :-1].unsqueeze(-2)
    a = real_dist.log_prob(observed_positions)
    b = spurious_dist.log_prob(observed_positions[..., :-1])
    log_likelihood = torch.cat((a, b), dim=-1)
    log_likelihood[~is_observed] = -float('inf')
    return log_likelihood


@poutine.broadcast
def guide(args, observations):
    positions = args.init_positions
    for em_iter in range(args.em_iters):
        positions = positions.detach()
        positions.requires_grad = True
        with torch.set_grad_enabled(False):
            assign_logits = compute_assign_logits(positions, observations, args)
            exists_logits = compute_exists_logits(positions, observations, args)
            assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                                      args.bp_iters,
                                                      bp_momentum=args.bp_momentum)
            p_exists = assignment.exists_dist.probs
            p_assign = assignment.assign_dist.probs

        log_likelihood = detection_log_likelihood(positions, observations, args)
        loss = -(log_likelihood * p_assign).sum()
        # M-step:
        positions, _ = newton_step(loss, positions, 5)
        if args.prune_threshold > 0.0:
            positions = positions[p_exists > args.prune_threshold]
        if args.merge_radius >= 0.0:
            positions, _ = merge_points(positions, args.merge_radius)

        warn_if_nan(positions, 'positions')

    assign_logits = compute_assign_logits(positions, observations, args)
    exists_logits = compute_exists_logits(positions, observations, args)
    assignment = MarginalAssignmentPersistent(exists_logits, assign_logits,
                                              args.bp_iters, bp_momentum=args.bp_momentum)
    return assignment, positions


def main(args):
    # Read file and create observations
    if args.download:
        download_mot17(args.mot17_dir)

    det_file = os.path.join(args.mot17_dir, args.sequence, 'det', 'det.txt')
    assert os.path.isfile(det_file), "{} doesn't exist".format(det_file)
    frames, object_id, positions, sizes, features = read_mot(det_file, zero_based=True)
    assert (object_id == -1).all(), object_id
    confidences = features['confidence']
    args.num_frames = max(frames).item() + 1
    args.max_detections_per_frame = torch.max(torch.histc(frames.float(),
                                                          bins=args.num_frames,
                                                          min=-.5, max=args.num_frames - .5))
    args.ymax = positions[..., 0].max() + sizes.max()
    args.xmax = positions[..., 1].max() + sizes.max()
    observations = torch.zeros(args.num_frames,
                               args.max_detections_per_frame, 4)  # 3 pos 1 confidence
    for i in range(args.num_frames):
        idx = frames == i
        num_detections = idx.sum()
        observations[i, :num_detections, :2] = positions[idx]
        observations[i, :num_detections, 2] = sizes[idx].mean(-1)
        observations[i, :num_detections, 3] = confidences[idx]

    # DEBUG initialize with correct positions
    if args.init_from_gt:
        gt_filename = os.path.join(args.mot17_dir, args.sequence, 'gt', 'gt.txt')
        args.init_positions = init_from_gt(gt_filename, args)
    else:
        args.init_positions = init_from_first_frame(frames, positions, sizes)

    # Inference:
    assignment, positions2 = guide(args, observations)
    assert ((positions2 - positions2[0]).abs() >= 1).any()

    # convert assignment to out_frames, out_object_id, out_positions, out_sizes, out_confidences
    sampled_assignment = assignment.assign_dist.sample()
    sampled_exists = assignment.exists_dist.sample()
    is_matched_detection = (observations[..., -1] > 0) & (sampled_assignment < args.max_num_objects)
    out_frames = torch.arange(args.num_frames).unsqueeze(-1).expand(is_matched_detection.shape)[is_matched_detection]
    out_object_id = sampled_assignment[is_matched_detection]
    out_positions = positions2[out_object_id, :2]
    out_sizes = positions2[out_object_id, 2].unsqueeze(-1).expand(-1, 2)
    out_confidences = sampled_exists[out_object_id]

    out_filename = os.path.join('./', args.sequence + '.txt')
    write_mot(out_filename, out_frames, out_object_id, out_positions,
              out_sizes, out_confidences)

    # Visualization
    if args.vis:
        import pymotutils
        from pymotutils.contrib.datasets import motchallenge
        devkit = motchallenge.Devkit(args.mot17_dir)
        data_source = devkit.create_data_source(args.sequence, 0)

        # Compute a suitable window shape.
        image_shape = data_source.peek_image_shape()[::-1]
        aspect_ratio = float(image_shape[0]) / image_shape[1]
        window_shape = int(aspect_ratio * 600), 600

        visualization = pymotutils.MonoVisualization(
            update_ms=data_source.update_ms, window_shape=window_shape)
        application = pymotutils.Application(data_source)
        hyp = pymotutils.motchallenge_io.read_groundtruth(out_filename, sensor_data_is_3d=False)
        application.play_track_set(hyp, visualization)


def parse_args(*args):
    from shlex import split
    parser = argparse.ArgumentParser()
    parser.add_argument('--mot17-dir', required=True)
    parser.add_argument('--sequence', default='MOT17-02-FRCNN')
    parser.add_argument('--init_from_gt', action="store_false", dest='init_from_gt', default=False,
                        help='Init from gt')
    parser.add_argument('--max-num-objects', default=400, type=int, help='maximum number of objects')
    parser.add_argument('--position-noise-scale', default=10, type=int, help='maximum number of objects')
    parser.add_argument('--bp-iters', default=50, type=int, help='number of BP iterations')
    parser.add_argument('--bp-momentum', default=0.5, type=float, help='BP momentum')
    parser.add_argument('--em-iters', default=10, type=int, help='number of EM iterations')
    parser.add_argument('--merge-radius', default=-1, type=float, help='merge radius')
    parser.add_argument('--prune-threshold', default=1e-3, type=float, help='prune threshold')
    parser.add_argument('--no-vis', action="store_false", dest='vis', default=True,
                        help='No visualization')
    parser.add_argument('--download', action="store_true", dest='download', default=False,
                        help='Download MOT17 dataset (~5.5GB)')
    if len(args):
        return parser.parse_args(split(args[0]))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
