"""
Schelling coordination game:
Two spies, Alice and Bob, want to meet.

They must choose between two locations without communicating
by recursively reasoning about one another.

Taken from: http://forestdb.org/models/schelling.html
"""

from __future__ import print_function

import torch
from torch.autograd import Variable

import pyro
from pyro.distributions.torch import bernoulli
from pyro.infer import Marginal, Search


def location(preference):
    """
    Flips a weighted coin to decide between two locations to meet
    """
    return pyro.sample("loc", bernoulli, preference)


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


# We sample Bob's choice of location by marginalizing
# over his decision process.
bob_rec = Marginal(Search(bob))

print(sum([bob_rec(Variable(torch.Tensor([0.6])), 2)
           for i in range(100)]) / 100.0)
