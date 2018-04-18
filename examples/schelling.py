"""
Schelling coordination game:
Two spies, Alice and Bob, want to meet.

They must choose between two locations without communicating
by recursively reasoning about one another.

Taken from: http://forestdb.org/models/schelling.html
"""

from __future__ import print_function

import argparse

import torch

import pyro
from pyro.distributions import Bernoulli
from pyro.infer import Marginal, Search


def location(preference):
    """
    Flips a weighted coin to decide between two locations to meet
    In this example, we assume that Alice and Bob share a prior preference
    for one location over another, reflected in the value of preference below.
    """
    return pyro.sample("loc", Bernoulli(preference))


def alice(preference, depth):
    """
    Alice decides where to go by reasoning about Bob's choice
    """
    alice_prior = location(preference)
    return pyro.sample("bob_choice", Marginal(Search(bob)),
                       preference, depth - 1,
                       obs=alice_prior)


def bob(preference, depth):
    """
    Bob decides where to go by reasoning about Alice's choice
    """
    bob_prior = location(preference)
    if depth > 0:
        return pyro.sample("alice_choice", Marginal(Search(alice)),
                           preference, depth,
                           obs=bob_prior)
    else:
        return bob_prior


def main(args):
    # We sample Bob's choice of location by marginalizing
    # over his decision process.
    bob_decision = Marginal(Search(bob))

    # Here Alice and Bob slightly prefer one location over the other a priori
    shared_preference = torch.tensor([args.preference])

    bob_depth = args.depth
    num_samples = args.num_samples

    # draw num_samples samples from Bob's decision process
    # and use those to estimate the marginal probability
    # that Bob chooses their preferred location
    bob_prob = sum([bob_decision(shared_preference, bob_depth)
                    for i in range(num_samples)]) / float(num_samples)

    print("Empirical frequency of Bob choosing their favored location " +
          "given preference {} and recursion depth {}: {}"
          .format(shared_preference, bob_depth, bob_prob))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-samples', default=10, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--preference', default=0.6, type=float)
    args = parser.parse_args()
    main(args)
