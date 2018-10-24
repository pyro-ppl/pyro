
from __future__ import absolute_import, division, print_function
import numpy as np

import argparse
import logging

import torch
from torch.distributions import constraints

import re
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoGuide, AutoGuideList, AutoDiagonalNormal
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, Trace_ELBO, JitTrace_ELBO, config_enumerate
from pyro.optim import Adam


logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)

def model_slds(sequences, lengths, args, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length

    n_state = args.num_state # K, number of states
    n_lat = args.num_latent
    n_out = data_dim
    normal_std = 1e-1
    with poutine.mask(mask=torch.tensor(include_prior)):
        # transition matrix for HMM, K*K,
        hmm_dyn = pyro.sample("hmm_dyn",
                              dist.Dirichlet(0.9 * torch.eye(n_state) + 0.1)
                                  .independent(1))
        ## LDS
        ssm_dyn = pyro.sample("ssm_dyn",
                              dist.Normal(0,normal_std).expand_by([n_state, n_lat, n_lat])
                                  .independent(3))
        ssm_bias = pyro.sample("ssm_bias",
                              dist.Normal(0,1.).expand_by([n_state, n_lat, 1])
                                  .independent(3))
        ssm_noise = pyro.sample("ssm_noise",
                              dist.Gamma(1e0,1e0).expand_by([n_state, n_lat, 1])
                                  .independent(3))
        ## observation
        obs_weight = pyro.sample("obs_weight",
                              dist.Normal(0,1).expand_by([n_state, n_out, n_lat])
                                  .independent(3))
        obs_bias = pyro.sample("obs_bias",
                              dist.Normal(0,1).expand_by([n_state, n_out, 1])
                                  .independent(3))
        obs_noise = pyro.sample("obs_noise",
                              dist.Gamma(1e0,1e0).expand_by([n_state, n_out, 1])
                                  .independent(3))

    with pyro.plate("sequences", len(sequences), dim=args.plate_dim) as batch:
        ## ssm init state
        # use a (n_lat,1) matrix (instead of vector) to ease batch operation,
        # as for high dim tensor , matmul would do batch matrix-matrix product, instead of matrix-vector product.
        #
        ssm_init = pyro.sample("ssm_init", dist.Normal(0,1).expand_by([n_lat,1]).independent(2))

        lengths = lengths[batch]
        z = 0
        x = ssm_init
        for t in range(lengths.max()):
            # with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
            z = pyro.sample("z_{}".format(t), dist.Categorical(hmm_dyn[z]),
                            infer={"enumerate": "parallel"})
            x = pyro.sample("x_{}".format(t), dist.Normal(torch.matmul(ssm_dyn[z], x) + ssm_bias[z], torch.sqrt(1./ssm_noise[z])).independent(2))
            obs_lat = torch.matmul(obs_weight[z], x) + obs_bias[z]
            pyro.sample("y_{}".format(t), dist.Normal(obs_lat, torch.sqrt(1./obs_noise[z])).independent(2),
                        obs=sequences[batch, t].unsqueeze(-1))


                # print('===={}===='.format(t))
                # print('obs_lat', obs_lat.shape)
                # print('obs_noise[z]', obs_noise[z].shape)
                # print('sequences[batch, t].unsqueeze(-1).shape', sequences[batch, t].unsqueeze(-1).shape)
                #
                # print('obs_bias[z]', obs_bias[z].shape)
                # print('obs_weight[z]', obs_weight[z].shape)
                #
                # print('z_shape:', z.shape)
                # print('hmm_dyn[z].shape', hmm_dyn[z].shape)
                # print('cat_event_shape', dist.Categorical(hmm_dyn[z]).event_shape)
                # print('cat_batch_shape', dist.Categorical(hmm_dyn[z]).batch_shape)
                #
                # print('x_shape:', x.shape)
                # print('x_batch_shape:', dist.Normal(torch.matmul(ssm_dyn[z], x) + ssm_bias[z], torch.sqrt(1. / ssm_noise[z])).independent(2).batch_shape)
                #
                # print('y_batch_shape', dist.Normal(obs_lat, torch.sqrt(1./obs_noise[z])).independent(2).batch_shape)
                # print('y_event.shape', dist.Normal(obs_lat, torch.sqrt(1./obs_noise[z])).independent(2).event_shape)
                # print('obs_final.shape', sequences[batch, t].unsqueeze(-1).shape)


def load_sequences(datafile, N=None):

    npz = np.load(datafile)
    data, labels, A1, A2, A3 = [npz['arr_%d' % i] for i in range(5)]

    # just repeat
    sequences = torch.tensor(np.array([data]*N), dtype=torch.float32) if N is not None else data
    return sequences

def gen_random_sequences(N,T,D):
    sequences = torch.tensor(np.random.randn(N,T,D), dtype=torch.float32)
    return sequences


def gen_nseg_ar_sequences(N,T,D,n_seg):
    # TODO
    return 0

def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Generating data')
    logging.info(args)

    # retfile = '/Users/ruijili/Documents/3_correlation/tsa/pao/sssm/ret.npz'
    retfile = 'slds_ret.npz'



    ########################################################################
    # just generate a random dataset for debug purpose
    ########################################################################
    N = 3  # number of sequences
    T = 10 # number of timestamps
    D = 2

    # change me
    if args.data_path != '':
        sequences = load_sequences(args.data_path, N)
    else:
        sequences = gen_random_sequences(N, T, D)


    N, T, D = sequences.shape
    lengths = torch.ones(N, dtype=torch.long)*T


    model = model_slds


    ###################
    # guide for MAP
    ###################
    guide_map = AutoGuideList(model)
    guide_map.add(AutoDelta(poutine.block(model, hide_fn=lambda m: re.match('[z]_', m['name']))))


    # guide_map.add(AutoDelta(poutine.block(model, hide_fn=lambda m: re.match('[zx]_', m['name']))))
    # guide_map.add(AutoDiagonalNormal(poutine.block(model, expose_fn=lambda m: re.match('x_', m['name']))))


    #################################################################
    # guide for VB on z, for simplicity of debug, focusing on z, ignoring the rest
    #################################################################
    guide_vb_0 = AutoGuideList(model)
    guide_vb_0.add(AutoDelta(poutine.block(model, hide_fn=lambda m: re.match('[zx]_', m['name']))))
    guide_vb_0.add(AutoDiagonalNormal(poutine.block(model, expose_fn=lambda m: re.match('x_', m['name']))))

    def guide_vb(sequences, lengths, args, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        # print('in guide, ', num_sequences, max_length, data_dim)
        n_state = args.num_state

        ####################
        ## if enabled there will be error claiming that 'sequence' is used twice.
        ########################
        # guide_vb_0(sequences, lengths, args, include_prior=True)

        with pyro.plate("sequences", len(sequences), dim=args.plate_dim) as batch:
            lengths = lengths[batch]
            for t in range(lengths.max()):
                ret_prob = pyro.param('assignment_probs_{}'.format(t), torch.ones(len(lengths), n_state) / n_state,
                                      constraint=constraints.simplex)
                z = pyro.sample("z_{}".format(t), dist.Categorical(ret_prob), infer={"enumerate": "parallel"})

        print('in guide, ', num_sequences, max_length, data_dim)
        print('in guide, z.shape', z.shape)
        print('in guide, max_lengths', max(lengths))



    guide = guide_map if args.map else guide_vb


    logging.info('Training {} on {} sequences'.format('slds', 1))
    if args.truncate:
        lengths.clamp_(max=args.truncate)
    num_observations = float(lengths.sum())
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(True)


    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.


    ######## debug purpose only, to check the shapes in mode and guide #####
    trace_model = poutine.trace(model).get_trace(sequences, lengths, args)
    trace_guide = poutine.trace(guide).get_trace(sequences, lengths, args)
    #####################





    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_plate_nesting=5)


    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences, lengths, args)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    # train_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False)
    # logging.info('training loss = {}'.format(train_loss / num_observations))



    ###############################
    ## exam the enumeration of z
    ###############################
    # save result for further investigation
    dict_ret = {}
    for name in pyro.get_param_store().get_all_param_names():
        dict_ret[name] = pyro.param(name).data.numpy()

    if args.map:
        logging.info("computing marginals")
        r1 = elbo.compute_marginals(model, guide, sequences, lengths, args)
        for k,v in r1.items():
            dict_ret['marg_%s' % k] = v.logits.data.numpy()

    np.savez(retfile, **dict_ret)


    logging.info('done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-n", "--num-steps", default=1, type=int)
    parser.add_argument("-l", "--num-latent", default=2, type=int)
    parser.add_argument("-s", "--num-state", default=5, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true', default=False)
    parser.add_argument("--map", action='store_true', default=False) # MAP estimation or VB, Use MAP for the moment as VB is not working
    parser.add_argument("-pd", "--plate-dim", default=-1, type=int)
    parser.add_argument("-dp", "--data-path", default='', type=str)
    # parser.add_argument("-dp", "--data-path", default='/Users/ruijili/Documents/3_correlation/tsa/pao/sssm/hmmar.npz', type=str)

    args = parser.parse_args()
    main(args)
