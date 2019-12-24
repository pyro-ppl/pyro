from collections import OrderedDict

from pyro.distributions.reparameterize import Reparameterizer
from pyro.infer.autoguide.guides import AutoContinuous


class NeuTra(Reparameterizer):
    """
    Example::

        neutra = NeuTra(AutoIAFNormal(model))
        model = poutine.infer_config(lambda _: {"reparam": neutra})
        model = poutine.reparam(model)
        # ...now use the model in HMC or NUTS...

    Note that all sites must share a common ``NeuTra`` instance.
    """
    def __init__(self, guide):
        assert isinstance(guide, AutoContinuous)
        self.guide
        self.posterior = None
        self.x_constrained = []

    def get_dists(self, fn):
        dists = OrderedDict()
        if not self.x_constrained:  # on first sample site
            self.posterior = self.guide.get_posterior()
            dists["latent"] = self.posterior
        return dists

    def transform_values(self, fn, values):
        if not self.x_constrainend:  # on first sample site
            x_unconstrained = values["latent"]
            x_constrained = self.guide._unpack_latent(x_unconstrained)
            self.x_constrained = list(reversed(x_constrained)))
        site, value = self.x_unconstrained.pop()
        return value

    def get_log_importance(self, fn, values, value):
        log_density = fn.log_prob(value)
        if values:  # on first sample site
            log_density = log_density - self.posterior.log_prob(values["latent"])
        return log_density
