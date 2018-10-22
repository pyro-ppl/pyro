
from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta, AutoGuide, AutoGuideList
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, Trace_ELBO, JitTrace_ELBO, config_enumerate
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)

@poutine.broadcast
def model_sssm(sequences, lengths, args, batch_size=None, include_prior=True):
    num_sequences, max_length, data_dim = sequences.shape
    assert lengths.shape == (num_sequences,)
    assert lengths.max() <= max_length

    n_seg = args.num_segment # K, number of segments
    n_lat = args.num_latent
    n_out = data_dim
    normal_std = 1e-1
    with poutine.mask(mask=torch.tensor(include_prior)):
        # transition matrix for HMM, K*K,
        hmm_dyn = pyro.sample("hmm_dyn",
                              dist.Dirichlet(0.9 * torch.eye(n_seg) + 0.1)
                                  .independent(1))
        ## SSM
        ssm_dyn = pyro.sample("ssm_dyn",
                              dist.Normal(0,normal_std).expand_by([n_seg, n_lat, n_lat])
                                  .independent(3))

        ssm_bias = pyro.sample("ssm_bias",
                              dist.Normal(0,1.).expand_by([n_seg, n_lat, 1])
                                  .independent(3))

        ssm_noise = pyro.sample("ssm_noise",
                              dist.Gamma(1e0,1e0).expand_by([n_seg, n_lat, 1])
                                  .independent(3))



        ## observation
        obs_weight = pyro.sample("obs_weight",
                              dist.Normal(0,1).expand_by([n_seg, n_out, n_lat])
                                  .independent(3))

        obs_bias = pyro.sample("obs_bias",
                              dist.Normal(0,1).expand_by([n_seg, n_out, 1])
                                  .independent(3))

        obs_noise = pyro.sample("obs_noise",
                              dist.Gamma(1e0,1e0).expand_by([n_seg, n_out, 1])
                                  .independent(3))



    with pyro.iarange("sequences", len(sequences), batch_size, dim=-1) as batch:
        ## ssm init state
        # use a (n_lat,1) matrix (instead of vector) to ease batch operation,
        # as for high dim tensor , matmul would do batch matrix-matrix product, instead of matrix-vector product.
        #
        ssm_init = pyro.sample("ssm_init", dist.Normal(0,1).expand_by([n_lat,1]).independent(2))

        lengths = lengths[batch]
        z = 0
        x = ssm_init
        for t in range(lengths.max()):
            with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                z = pyro.sample("z_{}".format(t), dist.Categorical(hmm_dyn[z]),
                                infer={"enumerate": "parallel"})
                x = pyro.sample("x_{}".format(t), dist.Normal(torch.matmul(ssm_dyn[z], x) + ssm_bias[z], torch.sqrt(1./ssm_noise[z])).independent(2))
                obs_lat = torch.matmul(obs_weight[z], x) + obs_bias[z]
                pyro.sample("y_{}".format(t), dist.Normal(obs_lat, torch.sqrt(1./obs_noise[z])).independent(2),
                            obs=sequences[batch, t].unsqueeze(-1))


                if 0:
                    print('===={}===='.format(t))
                    print('obs_lat', obs_lat.shape)
                    print('obs_noise[z]', obs_noise[z].shape)
                    print('sequences[batch, t].unsqueeze(-1).shape', sequences[batch, t].unsqueeze(-1).shape)

                    print('obs_bias[z]', obs_bias[z].shape)
                    print('obs_weight[z]', obs_weight[z].shape)

                    print('z_shape:', z.shape)
                    print('hmm_dyn[z].shape', hmm_dyn[z].shape)
                    print('cat_event_shape', dist.Categorical(hmm_dyn[z]).event_shape)
                    print('cat_batch_shape', dist.Categorical(hmm_dyn[z]).batch_shape)

                    print('x_shape:', x.shape)
                    print('x_batch_shape:', dist.Normal(torch.matmul(ssm_dyn[z], x) + ssm_bias[z], torch.sqrt(1. / ssm_noise[z])).independent(2).batch_shape)

                    print('y_batch_shape', dist.Normal(obs_lat, torch.sqrt(1./obs_noise[z])).independent(2).batch_shape)
                    print('y_event.shape', dist.Normal(obs_lat, torch.sqrt(1./obs_noise[z])).independent(2).event_shape)
                    print('obs_final.shape', sequences[batch, t].unsqueeze(-1).shape)

        # print('batch', batch)



def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')



    logging.info('Loading data')
    logging.info(args)

    import numpy as np
    datafile = '/Users/ruijili/Documents/3_correlation/tsa/pao/sssm/hmmar.npz'
    retfile = '/Users/ruijili/Documents/3_correlation/tsa/pao/sssm/ret.npz'

    npz = np.load(datafile)
    data, labels, A1, A2, A3 = [npz['arr_%d' % i] for i in range(5)]
    if args.truncate_data:
        data = data[:10,:]
    L = data.shape[0]
    logging.info('-' * 40)
    model = model_sssm


    ###################
    # fusion of guide
    ###################
    myguide_map = AutoDelta(poutine.block(model, hide_fn=lambda m: m['name'].startswith('z_')))
    print('autodelta guide done')

    # @poutine.broadcast
    @config_enumerate(default="parallel")
    def myguide_full(sequences, lengths, args, batch_size=None, include_prior=True):
        num_sequences, max_length, data_dim = sequences.shape
        # print('in guide, ', num_sequences, max_length, data_dim)
        n_seg = args.num_segment

        # myguide_map(sequences, lengths, args, batch_size, include_prior)
        with pyro.iarange("sequences", len(sequences), batch_size, dim=-2) as batch:
            lengths = lengths[batch]
            for t in range(lengths.max()):
                ret_prob = pyro.param('assignment_probs_{}'.format(t), torch.ones(len(lengths), 1, n_seg) / n_seg,
                                      constraint=constraints.unit_interval)

                # with poutine.mask(mask=(t < lengths).unsqueeze(-1)):
                z = pyro.sample("z_{}".format(t), dist.Categorical(ret_prob))
        # print('in guide, ', num_sequences, max_length, data_dim)
        # print('in guide, z.shape', z.shape)
        # print('in guide, max_lengths', max(lengths))

    ############ end of guide

    logging.info('Training {} on {} sequences'.format('sssm', 1))

    if args.batch_size==1:
        sequences = torch.tensor(data, dtype=torch.float32).unsqueeze(0) # reshape to (1, T, D_obs)
    else:
        sequences = torch.tensor(np.array([data]*args.batch_size), dtype=torch.float32)  # reshape to (3, T, D_obs)

    lengths = torch.ones(sequences.shape[0], dtype=torch.long)*L

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
    if args.map:
        guide = AutoDelta(poutine.block(model, hide_fn=lambda m: m['name'].startswith('z_')))

    else:
        guide = myguide_full




    ######## debug OF THE MODEL #####
    trace_model = poutine.trace(model).get_trace(sequences, lengths, args, batch_size=args.batch_size)
    trace_guide = poutine.trace(guide).get_trace(sequences, lengths, args, batch_size=args.batch_size)
    #####################



    # Enumeration requires a TraceEnum elbo and declaring the max_iarange_nesting.
    # All of our models have two iaranges: "data" and "tones".
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_iarange_nesting=2)


    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)
    # We'll train on small minibatches.
    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(sequences, lengths, args, batch_size=args.batch_size)
        logging.info('{: >5d}\t{}'.format(step, loss / num_observations))

    # We evaluate on the entire training dataset,
    # excluding the prior term so our results are comparable across models.
    train_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False)
    logging.info('training loss = {}'.format(train_loss / num_observations))



    ###############################
    ## exam the enumeration of z
    ###############################
    r1 = elbo.compute_marginals(model, guide, sequences, lengths, args, batch_size=args.batch_size)

    # save result for further investigation
    print('run guide for the last time')
    guide(sequences, lengths, args, batch_size=args.batch_size)
    dict_ret = {}
    import numpy as np
    for name in pyro.get_param_store().get_all_param_names():
        # logging.info(name)
        # logging.info(pyro.param(name).data.numpy())
        dict_ret[name] = pyro.param(name).data.numpy()

    for k,v in r1.items():
        dict_ret['marg_%s' % k] = v.logits.data.numpy()

    np.savez(retfile, **dict_ret)


    print('done examing enumeration')

    if 0:
    # Finally we evaluate on the test dataset.
        logging.info('-' * 40)
        logging.info('Evaluating on {} test sequences'.format(len(data['test']['sequences'])))
        sequences = torch.tensor(data['test']['sequences'], dtype=torch.float32)
        lengths = torch.tensor(data['test']['sequence_lengths'], dtype=torch.long)
        if args.truncate:
            lengths.clamp_(max=args.truncate)
        num_observations = float(lengths.sum())
        test_loss = elbo.loss(model, guide, sequences, lengths, args, include_prior=False)
        logging.info('test loss = {}'.format(test_loss / num_observations))

        # We expect models with higher capacity to perform better,
        # but eventually overfit to the training set.
        capacity = sum(len(pyro.param(name).reshape(-1))
                       for name in pyro.get_param_store().get_all_param_names())
        logging.info('{} capacity = {} parameters'.format(model.__name__, capacity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning Bach Chorales")
    parser.add_argument("-n", "--num-steps", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=3, type=int)
    parser.add_argument("-l", "--num-latent", default=2, type=int)
    parser.add_argument("-s", "--num-segment", default=5, type=int)

    parser.add_argument("-lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("-t", "--truncate", type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument("--map", action='store_true', default=True)
    parser.add_argument("--truncate-data", action='store_true', default=True)

    args = parser.parse_args()
    main(args)
