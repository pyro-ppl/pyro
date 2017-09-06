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

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(ScalePoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(ScalePoutine, self)._pyro_observe(msg, name, fn, obs, *args, **kwargs)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None):
        """
        Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(ScalePoutine, self)._pyro_map_data(msg, name, data, fn,
                                                        batch_size=batch_size)
