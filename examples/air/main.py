"""
AIR applied to the multi-mnist data set [1].

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""
from __future__ import division

import argparse
import math
import os
import time
from functools import partial

import numpy as np
import torch
import visdom
from observations import multi_mnist

import pyro
import pyro.optim as optim
import pyro.poutine as poutine
from air import AIR, latents_to_tensor
from pyro.infer import SVI, TraceGraph_ELBO
from viz import draw_many, tensor_to_objs


def count_accuracy(X, true_counts, air, batch_size):
    assert X.size(0) == true_counts.size(0), 'Size mismatch.'
    assert X.size(0) % batch_size == 0, 'Input size must be multiple of batch_size.'
    counts = torch.LongTensor(3, 4).zero_()
    error_latents = []
    error_indicators = []

    def count_vec_to_mat(vec, max_index):
        out = torch.LongTensor(vec.size(0), max_index + 1).zero_()
        out.scatter_(1, vec.type(torch.LongTensor).view(vec.size(0), 1), 1)
        return out

    for i in range(X.size(0) // batch_size):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        true_counts_batch = true_counts[i * batch_size:(i + 1) * batch_size]
        z_where, z_pres = air.guide(X_batch, batch_size)
        inferred_counts = sum(z.cpu() for z in z_pres).squeeze().data
        true_counts_m = count_vec_to_mat(true_counts_batch, 2)
        inferred_counts_m = count_vec_to_mat(inferred_counts, 3)
        counts += torch.mm(true_counts_m.t(), inferred_counts_m)
        error_ind = 1 - (true_counts_batch == inferred_counts)
        error_ix = error_ind.nonzero().squeeze()
        error_latents.append(latents_to_tensor((z_where, z_pres)).index_select(0, error_ix))
        error_indicators.append(error_ind)

    acc = counts.diag().sum().float() / X.size(0)
    error_indices = torch.cat(error_indicators).nonzero().squeeze()
    if X.is_cuda:
        error_indices = error_indices.cuda()
    return acc, counts, torch.cat(error_latents), error_indices


# Defines something like a truncated geometric. Like the geometric,
# this has the property that there's a constant difference in log prob
# between p(steps=n) and p(steps=n+1).
def make_prior(k):
    assert 0 < k <= 1
    u = 1 / (1 + k + k**2 + k**3)
    p0 = 1 - u
    p1 = 1 - (k * u) / p0
    p2 = 1 - (k**2 * u) / (p0 * p1)
    trial_probs = [p0, p1, p2]
    # dist = [1 - p0, p0 * (1 - p1), p0 * p1 * (1 - p2), p0 * p1 * p2]
    # print(dist)
    return lambda t: trial_probs[t]


# Implements "prior annealing" as described in this blog post:
# http://akosiorek.github.io/ml/2017/09/03/implementing-air.html

# That implementation does something very close to the following:
# --z-pres-prior (1 - 1e-15)
# --z-pres-prior-raw
# --anneal-prior exp
# --anneal-prior-to 1e-7
# --anneal-prior-begin 1000
# --anneal-prior-duration 1e6

# e.g. After 200K steps z_pres_p will have decayed to ~0.04

# These compute the value of a decaying value at time t.
# initial: initial value
# final: final value, reached after begin + duration steps
# begin: number of steps before decay begins
# duration: number of steps over which decay occurs
# t: current time step


def lin_decay(initial, final, begin, duration, t):
    assert duration > 0
    x = (final - initial) * (t - begin) / duration + initial
    return max(min(x, initial), final)


def exp_decay(initial, final, begin, duration, t):
    assert final > 0
    assert duration > 0
    # half_life = math.log(2) / math.log(initial / final) * duration
    decay_rate = math.log(initial / final) / duration
    x = initial * math.exp(-decay_rate * (t - begin))
    return max(min(x, initial), final)


def load_data():
    inpath = './data'
    (X_np, Y), _ = multi_mnist(inpath, max_digits=2, canvas_size=50, seed=42)
    X_np = X_np.astype(np.float32)
    X_np /= 255.0
    X = torch.from_numpy(X_np)
    # Using FloatTensor to allow comparison with values sampled from
    # Bernoulli.
    counts = torch.FloatTensor([len(objs) for objs in Y])
    return X, counts


def main(**kwargs):

    args = argparse.Namespace(**kwargs)

    if 'save' in args:
        if os.path.exists(args.save):
            raise RuntimeError('Output file "{}" already exists.'.format(args.save))

    if args.seed is not None:
        pyro.set_rng_seed(args.seed)

    X, true_counts = load_data()
    X_size = X.size(0)
    if args.cuda:
        X = X.cuda()

    # Build a function to compute z_pres prior probabilities.
    if args.z_pres_prior_raw:
        def base_z_pres_prior_p(t):
            return args.z_pres_prior
    else:
        base_z_pres_prior_p = make_prior(args.z_pres_prior)

    # Wrap with logic to apply any annealing.
    def z_pres_prior_p(opt_step, time_step):
        p = base_z_pres_prior_p(time_step)
        if args.anneal_prior == 'none':
            return p
        else:
            decay = dict(lin=lin_decay, exp=exp_decay)[args.anneal_prior]
            return decay(p, args.anneal_prior_to, args.anneal_prior_begin,
                         args.anneal_prior_duration, opt_step)

    model_arg_keys = ['window_size',
                      'rnn_hidden_size',
                      'decoder_output_bias',
                      'decoder_output_use_sigmoid',
                      'baseline_scalar',
                      'encoder_net',
                      'decoder_net',
                      'predict_net',
                      'embed_net',
                      'bl_predict_net',
                      'non_linearity',
                      'pos_prior_mean',
                      'pos_prior_sd',
                      'scale_prior_mean',
                      'scale_prior_sd']
    model_args = {key: getattr(args, key) for key in model_arg_keys if key in args}
    air = AIR(
        num_steps=args.model_steps,
        x_size=50,
        use_masking=not args.no_masking,
        use_baselines=not args.no_baselines,
        z_what_size=args.encoder_latent_size,
        use_cuda=args.cuda,
        **model_args
    )

    if args.verbose:
        print(air)
        print(args)

    if 'load' in args:
        print('Loading parameters...')
        air.load_state_dict(torch.load(args.load))

    vis = visdom.Visdom(env=args.visdom_env)
    # Viz sample from prior.
    if args.viz:
        z, x = air.prior(5, z_pres_prior_p=partial(z_pres_prior_p, 0))
        vis.images(draw_many(x, tensor_to_objs(latents_to_tensor(z))))

    def per_param_optim_args(module_name, param_name):
        lr = args.baseline_learning_rate if 'bl_' in param_name else args.learning_rate
        return {'lr': lr}

    svi = SVI(air.model, air.guide,
              optim.Adam(per_param_optim_args),
              loss=TraceGraph_ELBO())

    # Do inference.
    t0 = time.time()
    examples_to_viz = X[5:10]

    for i in range(1, args.num_steps + 1):

        loss = svi.step(X, args.batch_size, z_pres_prior_p=partial(z_pres_prior_p, i))

        if args.progress_every > 0 and i % args.progress_every == 0:
            print('i={}, epochs={:.2f}, elapsed={:.2f}, elbo={:.2f}'.format(
                i,
                (i * args.batch_size) / X_size,
                (time.time() - t0) / 3600,
                loss / X_size))

        if args.viz and i % args.viz_every == 0:
            trace = poutine.trace(air.guide).get_trace(examples_to_viz, None)
            z, recons = poutine.replay(air.prior, trace=trace)(examples_to_viz.size(0))
            z_wheres = tensor_to_objs(latents_to_tensor(z))

            # Show data with inferred objection positions.
            vis.images(draw_many(examples_to_viz, z_wheres))
            # Show reconstructions of data.
            vis.images(draw_many(recons, z_wheres))

        if args.eval_every > 0 and i % args.eval_every == 0:
            # Measure accuracy on subset of training data.
            acc, counts, error_z, error_ix = count_accuracy(X, true_counts, air, 1000)
            print('i={}, accuracy={}, counts={}'.format(i, acc, counts.numpy().tolist()))
            if args.viz and error_ix.size(0) > 0:
                vis.images(draw_many(X[error_ix[0:5]], tensor_to_objs(error_z[0:5])),
                           opts=dict(caption='errors ({})'.format(i)))

        if 'save' in args and i % args.save_every == 0:
            print('Saving parameters...')
            torch.save(air.state_dict(), args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pyro AIR example", argument_default=argparse.SUPPRESS)
    parser.add_argument('-n', '--num-steps', type=int, default=int(1e8),
                        help='number of optimization steps to take')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='batch size')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('-blr', '--baseline-learning-rate', type=float, default=1e-3,
                        help='baseline learning rate')
    parser.add_argument('--progress-every', type=int, default=1,
                        help='number of steps between writing progress to stdout')
    parser.add_argument('--eval-every', type=int, default=0,
                        help='number of steps between evaluations')
    parser.add_argument('--baseline-scalar', type=float,
                        help='scale the output of the baseline nets by this value')
    parser.add_argument('--no-baselines', action='store_true', default=False,
                        help='do not use data dependent baselines')
    parser.add_argument('--encoder-net', type=int, nargs='+', default=[200],
                        help='encoder net hidden layer sizes')
    parser.add_argument('--decoder-net', type=int, nargs='+', default=[200],
                        help='decoder net hidden layer sizes')
    parser.add_argument('--predict-net', type=int, nargs='+',
                        help='predict net hidden layer sizes')
    parser.add_argument('--embed-net', type=int, nargs='+',
                        help='embed net architecture')
    parser.add_argument('--bl-predict-net', type=int, nargs='+',
                        help='baseline predict net hidden layer sizes')
    parser.add_argument('--non-linearity', type=str,
                        help='non linearity to use throughout')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='generate vizualizations during optimization')
    parser.add_argument('--viz-every', type=int, default=100,
                        help='number of steps between vizualizations')
    parser.add_argument('--visdom-env', default='main',
                        help='visdom enviroment name')
    parser.add_argument('--load', type=str,
                        help='load previously saved parameters')
    parser.add_argument('--save', type=str,
                        help='save parameters to specified file')
    parser.add_argument('--save-every', type=int, default=1e4,
                        help='number of steps between parameter saves')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda')
    parser.add_argument('-t', '--model-steps', type=int, default=3,
                        help='number of time steps')
    parser.add_argument('--rnn-hidden-size', type=int, default=256,
                        help='rnn hidden size')
    parser.add_argument('--encoder-latent-size', type=int, default=50,
                        help='attention window encoder/decoder latent space size')
    parser.add_argument('--decoder-output-bias', type=float,
                        help='bias added to decoder output (prior to applying non-linearity)')
    parser.add_argument('--decoder-output-use-sigmoid', action='store_true',
                        help='apply sigmoid function to output of decoder network')
    parser.add_argument('--window-size', type=int, default=28,
                        help='attention window size')
    parser.add_argument('--z-pres-prior', type=float, default=0.5,
                        help='prior success probability for z_pres')
    parser.add_argument('--z-pres-prior-raw', action='store_true', default=False,
                        help='use --z-pres-prior directly as success prob instead of a geometric like prior')
    parser.add_argument('--anneal-prior', choices='none lin exp'.split(), default='none',
                        help='anneal z_pres prior during optimization')
    parser.add_argument('--anneal-prior-to', type=float, default=1e-7,
                        help='target z_pres prior prob')
    parser.add_argument('--anneal-prior-begin', type=int, default=0,
                        help='number of steps to wait before beginning to anneal the prior')
    parser.add_argument('--anneal-prior-duration', type=int, default=100000,
                        help='number of steps over which to anneal the prior')
    parser.add_argument('--pos-prior-mean', type=float,
                        help='mean of the window position prior')
    parser.add_argument('--pos-prior-sd', type=float,
                        help='std. dev. of the window position prior')
    parser.add_argument('--scale-prior-mean', type=float,
                        help='mean of the window scale prior')
    parser.add_argument('--scale-prior-sd', type=float,
                        help='std. dev. of the window scale prior')
    parser.add_argument('--no-masking', action='store_true', default=False,
                        help='do not mask out the costs of unused choices')
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='write hyper parameters and network architecture to stdout')
    main(**vars(parser.parse_args()))
