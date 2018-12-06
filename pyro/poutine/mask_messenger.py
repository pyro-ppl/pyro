from __future__ import absolute_import, division, print_function

import torch

from pyro.util import ignore_jit_warnings

from .messenger import Messenger


class MaskMessenger(Messenger):
    """
    This messenger masks sample sites.

    This is typically used for masking out parts of tensors.

    :param torch.ByteTensor mask: a ``{0,1}``-valued masking tensor
        (1 includes a site, 0 excludes a site)
    """
    def __init__(self, mask):
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.uint8:
                raise ValueError('Expected mask to be a ByteTensor but got {}'.format(type(mask)))
        else:
            if mask not in (True, False):
                raise ValueError('Expected mask to be a boolean but got {}'.format(type(mask)))
            with ignore_jit_warnings():
                mask = torch.tensor(mask)
        super(MaskMessenger, self).__init__()
        self.mask = mask

    def _process_message(self, msg):
        msg["mask"] = self.mask if msg["mask"] is None else self.mask & msg["mask"]
        return None
