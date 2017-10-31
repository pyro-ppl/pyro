"""
This attempts (currently unsuccessfully) to reproduce some of the
results on the multi-mnist data set described in [1].

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

import os
import sys
import time
import argparse
from functools import partial
from subprocess import check_call
import numpy as np

import torch
from torch.autograd import Variable

import pyro
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.infer import SVI

import visdom

from air import AIR
from viz import draw_many, post_process_latents

parser = argparse.ArgumentParser(description="Pyro AIR example", argument_default=argparse.SUPPRESS)
parser.add_argument('-n', '--num-steps', type=int, default=int(1e8),
                    help='number of optimization steps to take')
parser.add_argument('-b', '--batch-size', type=int, default=64,
                    help='batch size')
parser.add_argument('--progress-every', type=int, default=1,
                    help='number of steps between writing progress to stdout')
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
# parser.add_argument('--checkpoint', action='store_true',
#                     help='periodically persist parameters')
# parser.add_argument('--checkpoint-every', type=int, default=1000,
#                     help='number of steps between checkpoints')
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
parser.add_argument('--window-size', type=int, default=28,
                    help='attention window size')
parser.add_argument('--z-pres-prior', type=float, default=None,
                    help='prior success probability for z_pres')
parser.add_argument('--anneal-prior', action='store_true', default=False,
                    help='anneal z_pres prior during optimization')
parser.add_argument('--anneal-prior-from', type=float, default=0.99,
                    help='initial z_pres prior prob')
parser.add_argument('--anneal-prior-over', type=int, default=100000,
                    help='number of steps over which to anneal the prior')
parser.add_argument('--no-masking', action='store_true', default=False,
                    help='do not mask out the costs of unused choices')
parser.add_argument('--fudge-z-pres', action='store_true', default=False,
                    help='fudge z_pres to remove discreteness for testing')
parser.add_argument('--seed', type=int, help='random seed', default=None)
parser.add_argument('--print-modules', action='store_true',
                    help='write the network architecture to stdout')

args = parser.parse_args()
# print(args)

if args.seed is not None:
    pyro.set_rng_seed(args.seed)

# Load data.
infile = './data/multi_mnist_train_uint8.npz'
if not os.path.exists(infile):
    print('Running multi_mnist.py to generate dataset at {}...'.format(infile))
    multi_mnist_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multi_mnist.py')
    check_call([sys.executable, multi_mnist_py])
    print('Finished running multi_mnist.py.')
X_np = np.load(infile)['x'].astype(np.float32)
X_np /= 255.0
X = Variable(torch.from_numpy(X_np))
X_size = X.size(0)
if args.cuda:
    X = X.cuda()


# Yields the following distribution over the number of steps (when
# taking a maximum of 3 steps):
# p(0) = 0.4
# p(1) = 0.3
# p(2) = 0.2
# p(3) = 0.1
def default_z_pres_prior_p(t):
    if t == 0:
        return 0.6
    elif t == 1:
        return 0.5
    else:
        return 0.33


# Implements "prior annealing" as described in this blog post:
# http://akosiorek.github.io/ml/2017/09/03/implementing-air.html
def annealed_z_pres_prior_p(opt_step, time_step):
    p_0 = args.anneal_prior_from
    p_final = args.z_pres_prior or default_z_pres_prior_p(time_step)
    s = min(opt_step / args.anneal_prior_over, 1.0)
    return p_final * s + p_0 * (1 - s)


def z_pres_prior_p(opt_step, time_step):
    if args.anneal_prior:
        return annealed_z_pres_prior_p(opt_step, time_step)
    elif args.z_pres_prior:
        return args.z_pres_prior
    else:
        return default_z_pres_prior_p(time_step)


model_arg_keys = ['window_size',
                  'rnn_hidden_size',
                  'decoder_output_bias',
                  'baseline_scalar',
                  'encoder_net',
                  'decoder_net',
                  'predict_net',
                  'embed_net',
                  'bl_predict_net',
                  'non_linearity',
                  'fudge_z_pres',
                  'print_modules']
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

vis = visdom.Visdom(env=args.visdom_env)
# Viz sample from prior.
if args.viz:
    z, x = air.prior(5, z_pres_prior_p=partial(z_pres_prior_p, 0))
    vis.images(draw_many(x, post_process_latents(z)))

t0 = time.time()
examples_to_viz = X[9:14]


# Do inference.
def per_param_optim_args(module_name, param_name, tags):
    lr = 1e-3 if 'baseline' in tags else 1e-4
    return {'lr': lr}


svi = SVI(air.model, air.guide,
          optim.Adam(per_param_optim_args),
          loss='ELBO',
          trace_graph=True)

for i in range(args.num_steps):

    loss = svi.step(X, args.batch_size, z_pres_prior_p=partial(z_pres_prior_p, i))

    if i % args.progress_every == 0:
        print('i={}, epochs={:.2f}, elapsed={:.2f}, elbo={:.2f}'.format(
            i,
            (i * args.batch_size) / X_size,
            (time.time() - t0) / 3600,
            loss / X_size))

    if args.viz and (i + 1) % args.viz_every == 0:
        trace = poutine.trace(air.guide).get_trace(examples_to_viz, None)
        z, recons = poutine.replay(air.prior, trace)(examples_to_viz.size(0))
        z_wheres = post_process_latents(z)

        # Show data with inferred objection positions.
        vis.images(draw_many(examples_to_viz, z_wheres))
        # Show reconstructions of data.
        vis.images(draw_many(recons, z_wheres))

        # TODO: Report accuracy on predictions of object counts.
