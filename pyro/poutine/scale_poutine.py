from __future__ import absolute_import, division, print_function

from .poutine import Messenger, Poutine


class ScaleMessenger(Messenger):
    """
    This messenger rescales the log probability score.

    This is typically used for data subsampling or for stratified sampling of data
    (e.g. in fraud detection where negatives vastly outnumber positives).

    :param scale: a positive scaling factor
    :type scale: float or torch.autograd.Variable
    """
    def __init__(self, scale):
        super(ScaleMessenger, self).__init__()
        self.scale = scale

    def _prepare_site(self, msg):
        msg["scale"] = self.scale * msg["scale"]
        return msg


class ScalePoutine(Poutine):
    """
    This poutine rescales the log probability score.

    This is typically used for data subsampling or for stratified sampling of data
    (e.g. in fraud detection where negatives vastly outnumber positives).

    :param fn: an optional function to be scaled
    :type fn: callable or None
    :param scale: a positive scaling factor
    :type scale: float or torch.autograd.Variable
    """
    def __init__(self, fn, scale):
        super(ScalePoutine, self).__init__(fn)
        self.msngr = ScaleMessenger(scale)

    def _prepare_site(self, msg):
        return self.msngr._prepare_site(msg)
