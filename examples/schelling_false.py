"""
Schelling coordination game with false belief:
Two agents, Alice and Bob, claim to want to meet.
Bob wants to meet Alice, but Alice actually wants to avoid Bob.

They choose between two locations
by recursively reasoning about one another.

Taken from: http://forestdb.org/models/schelling-falsebelief.html
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


def alice_fb(preference, depth):
    """
    Alice's actual decision process:
    Alice decides where to go by reasoning about Bob's choice
    and choosing the other location.
    """
    alice_prior = location(preference)
    pyro.sample("bob_choice", Marginal(Search(bob)),
                preference, depth - 1,
                obs=alice_prior)
    return 1 - alice_prior


def alice(preference, depth):
    """
    Bob's model of Alice:
    Alice decides where to go by reasoning about Bob's choice
    and choosing that location
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


# We sample Alice's true choice of location
# by marginalizing over her decision process
alice_decision = Marginal(Search(alice_fb))

print(sum([alice_decision(Variable(torch.Tensor([0.55])), 3)
           for i in range(10)]))
