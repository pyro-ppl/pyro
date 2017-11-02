from __future__ import absolute_import, division, print_function

import torch.nn as nn


class ClippedSoftmax(nn.Softmax):
    """
    a wrapper around `nn.Softmax` that scales its output
    from `[0,1]` to `[epsilon,1-(n-1)*epsilon]`
    where n is the output dimension of Softmax
    """
    def __init__(self, epsilon, *args, **kwargs):
        self.epsilon = epsilon
        super(ClippedSoftmax, self).__init__(*args, **kwargs)

    def forward(self, val):
        rval = super(ClippedSoftmax, self).forward(val)
        n = rval.size(getattr(self, "dim", -1))
        return (rval * (1.0 - n * self.epsilon)) + self.epsilon


class ClippedSigmoid(nn.Sigmoid):
    """
    a wrapper around `nn.Sigmoid` that scales its output
    from `[0,1]` to `[epsilon,1-epsilon]`
    """
    def __init__(self, epsilon, *args, **kwargs):
        self.epsilon = epsilon
        super(ClippedSigmoid, self).__init__(*args, **kwargs)

    def forward(self, val):
        rval = super(ClippedSigmoid, self).forward(val)
        return (rval * (1.0 - 2 * self.epsilon)) + self.epsilon
