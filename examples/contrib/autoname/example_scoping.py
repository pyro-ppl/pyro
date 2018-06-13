import argparse

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.contrib.autoname import scope


# in this example, we'll see how scoping and non-strict naming
# makes Pyro programs significantly more composable

def model():
    pass


def submodel():
    pass


def main(args):
    pass


if __name__ == "__main__":
    main(arg)
