from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)


@poutine.broadcast
def model(args, data=None):
    # Globals.
    with pyro.iarange("topics", args.num_topics):
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / args.num_topics, 1.))
        topic_words = pyro.sample("topic_words",
                                  dist.Dirichlet(torch.ones(args.num_words) / args.num_words))

    # Locals.
    with pyro.iarange("documents", args.num_docs) as ind:
        if data is not None:
            assert data.shape == (args.num_words_per_doc, args.num_docs)
            data = data[:, ind]
        doc_topics = pyro.sample("doc_topics", dist.Dirichlet(topic_weights))
        with pyro.iarange("words", args.num_words_per_doc):
            word_topics = pyro.sample("word_topics", dist.Categorical(doc_topics),
                                      infer={"enumerate": "parallel"})
            # TODO use poutine.mask if docs have different lengths
            data = pyro.sample("doc_words", dist.Categorical(topic_words[word_topics]),
                               obs=data)

    return topic_weights, topic_words, data


class Guide(nn.Module):
    def __init__(self, args):
        super(Guide, self).__init__()
        layer_sizes = ([args.num_words] +
                       [int(s) for s in args.layer_sizes.split('-')] +
                       [args.num_topics])
        logging.info('Creating MLP with sizes {}'.format(layer_sizes))
        layers = []
        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layer = nn.Linear(in_size, out_size)
            layer.weight.data.normal_(0, 0.001)
            layer.bias.data.normal_(0, 0.001)
            layers.append(layer)
            layers.append(nn.Sigmoid())
        self.local_guide = nn.Sequential(*layers)

    @poutine.broadcast
    def __call__(self, args, data):
        # Use a conjugate guide for global variables.
        topic_weights_posterior = pyro.param(
                "topic_weights_posterior",
                lambda: torch.ones(args.num_topics) / args.num_topics,
                constraint=constraints.positive)
        topic_words_posterior = pyro.param(
                "topic_words_posterior",
                lambda: torch.ones(args.num_topics, args.num_words) / args.num_words,
                constraint=constraints.positive)
        with pyro.iarange("topics", args.num_topics):
            pyro.sample("topic_weights", dist.Gamma(topic_weights_posterior, 1.))
            pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

        # Use an amortized guide for local variables.
        pyro.module("local_guide", self.local_guide)
        with pyro.iarange("documents", args.num_docs, args.batch_size) as ind:
            counts = torch.zeros(args.num_words, args.batch_size)
            counts.scatter_add_(0, data[:, ind], torch.tensor(1.).expand(counts.shape))
            doc_topics = self.local_guide(counts.transpose(0, 1))
            pyro.sample("doc_topics", dist.Delta(doc_topics, event_dim=1))


def main(args):
    logging.info('Generating data')
    pyro.set_rng_seed(0)
    true_topic_weights, true_topic_words, data = model(args)

    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(args.num_docs))
    guide = Guide(args)
    Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    elbo = Elbo(max_iarange_nesting=2)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    logging.info('Step\tLoss')
    for step in range(args.num_steps):
        loss = svi.step(args, data)
        if step % 10 == 0:
            logging.info('{: >5d}\t{}'.format(step, loss))
    loss = elbo.loss(model, guide, args, data)
    logging.info('final loss = {}'.format(loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-t", "--num-topics", default=8, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-wd", "--num-words-per-doc", default=32, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    parser.add_argument("-lr", "--learning-rate", default=0.001, type=float)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
