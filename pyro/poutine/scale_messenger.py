from __future__ import absolute_import, division, print_function

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
        super(ScaleMessenger, self).__init__()
        self.scale = scale

    def _process_message(self, msg):
        msg["scale"] = self.scale * msg["scale"]
        return None
