from __future__ import absolute_import, division, print_function

from .poutine import Poutine


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
        self.scale = scale
        super(ScalePoutine, self).__init__(fn)

    def _prepare_site(self, msg):
        msg["scale"] = self.scale * msg["scale"]
        return msg
