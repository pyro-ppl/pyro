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
        Constructor: basically default, but store an extra scalar self.scale
        """
        self.scale = scale
        super(ScalePoutine, self).__init__(fn)

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Scaled sampling: Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(ScalePoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        Scaled observe: Rescale the message and continue
        """
        msg["scale"] = self.scale * msg["scale"]
        return super(ScalePoutine, self)._pyro_observe(msg, name, fn, obs, *args, **kwargs)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None, batch_dim=0):
        """
        Scaled map_data: Rescale the message and continue
        Should just work...
        """
        mapdata_scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        return super(ScalePoutine, self)._pyro_map_data(msg, name, data,
                                                        ScalePoutine(fn, mapdata_scale),
                                                        batch_size=batch_size,
                                                        batch_dim=batch_dim)
