from __future__ import absolute_import, division, print_function

import torch.nn.functional as F

import pyro
import pyro.distributions as dist

from .likelihood import Likelihood


def _softmax(x):
    return F.softmax(x, dim=-1)


class MultiClass(Likelihood):
    """
    Implementation of MultiClass likelihood, which is used for multi-class classification.

    :param int num_classes: Number of classes for prediction.
    :param callable response_function: A mapping to correct domain for MultiClass likelihood.
        By default, we use `softmax` function.
    """
    def __init__(self, num_classes, response_function=None):
        super(MultiClass, self).__init__()
        self.num_classes = num_classes
        self.response_function = (response_function if response_function is not None
                                  else _softmax)

    def forward(self, f, y=None):
        if f.dim() < 2:
            raise ValueError("Latent function output should be at least 2 dimensional: "
                             "one for number of classes and one for number of data.")
        # swap class dimension and data dimension
        f_swap = f.transpose(-2, -1)  # -> num_data x num_classes
        if f.size(-1) != self.num_classes:
            raise ValueError("Number of Gaussian processes should be equal to the "
                             "number of classes. Expected {} but got {}."
                             .format(self.num_classes, f.size(-1)))
        f_res = self.response_function(f_swap)
        if y is None:
            return pyro.sample("y", dist.Categorical(f_res))
        else:
            f_shape = y.size() + f_swap.size()[-1:]
            return pyro.sample("y", dist.Categorical(f_res.expand(f_shape))
                               .reshape(extra_event_dims=y.dim()), obs=y)
