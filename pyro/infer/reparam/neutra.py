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

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704
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
            x_unconstrained = pyro.sample("{}_latent".format(name), posterior)
            log_density = -posterior.log_prob(x_unconstrained)
            x_constrained = self.guide._unpack_latent(x_unconstrained)
            self.x_constrained = list(reversed(x_constrained))

        # Extract a single site's value from the shared latent.
        site, value = self.x_unconstrained.pop()
        log_density = log_density + fn.log_prob(value)
        new_fn = dist.Delta(value, log_density, event_dim=fn.event_dim)
        return new_fn, value
