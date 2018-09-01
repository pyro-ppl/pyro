from __future__ import absolute_import, division, print_function

import torch

from pyro.poutine.util import is_validation_enabled

from .messenger import Messenger


class ScaleMessenger(Messenger):
    """
    This messenger rescales the log probability score.

    This is typically used for data subsampling or for stratified sampling of data
    (e.g. in fraud detection where negatives vastly outnumber positives).

    :param scale: a positive scaling factor
    :type scale: float or torch.Tensor
    """
    def __init__(self, scale):
        if isinstance(scale, torch.Tensor):
            if is_validation_enabled() and not (scale > 0).all():
                raise ValueError("Expected scale > 0 but got {}. ".format(scale) +
                                 "Consider using poutine.mask() instead of poutine.scale().")
        elif not (scale > 0):
            raise ValueError("Expected scale > 0 but got {}".format(scale))
        super(ScaleMessenger, self).__init__()
        self.scale = scale

    def _process_message(self, msg):
        msg["scale"] = self.scale * msg["scale"]
        return None
