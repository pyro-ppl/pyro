import pyro
import pyro.distributions as dist
from pyro.infer.autoguide.guides import AutoContinuous


class NeuTraReparam:
    """
    Example::

        neutra = NeuTraReparam(AutoIAFNormal(model))
        model = poutine.infer_config(lambda _: {"reparam": neutra})
        model = poutine.reparam(model)
        # ...now use the model in HMC or NUTS...

    Note that all sites must share a common ``NeuTra`` instance.
    """
    def __init__(self, guide):
        assert isinstance(guide, AutoContinuous)
        self.guide
        self.x_constrained = []

    def __call__(self, name, fn, obs):
        log_density = 0.
        if not self.x_constrained:
            # Sample a shared latent on first sample site.
            posterior = self.guide.get_posterior()
            latent = pyro.sample("{}_latent".format(name), posterior)
            log_density = -posterior.log_prob(latent)
            x_constrained = self.guide._unpack_latent(x_unconstrained)
            self.x_constrained = list(reversed(x_constrained))

        # Extract a single site's value from the shared latent.
        site, value = self.x_unconstrained.pop()
        log_density = log_density + fn.log_prob(value)
        new_fn = dist.Delta(value, log_density, event_dim=fn.event_dim)
        return new_fn, value
