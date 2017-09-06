import pyro

from .poutine import Poutine


class ScalePoutine(Poutine):
    """
    score-rescaling Poutine
    Subsampling means we have to rescale pdfs inside map_data
    This poutine handles the rescaling because it wouldn't fit in Poutine
    """
    def __init__(self, fn, scale):
        """
        Constructor
        """
        self.scale = scale
        super(ScalePoutine, self).__init__(fn)

    def down(self, msg):
        """
        ScalePoutine has a side effect - pass the scale down the stack via msg
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(ScalePoutine, self).down(msg)
