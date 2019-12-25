import pyro
import pyro.distributions as dist
from pyro.infer.autoguide.guides import AutoContinuous


class NeuTraReparam:
    """
    Neural Transport reparameterizer [1] of multiple latent sites.

    This uses a trained :class:`~pyro.infer.autoguide.AutoContinuous`
    guide to alter the geometry of a model, typically for use e.g. in MCMC.
    Example usage::

        # Step 1. Train a guide
        guide = AutoIAFNormal(model)
        svi = SVI(model, guide, ...)
        # ...train the guide...

        # Step 2. Use trained guide in NeuTra MCMC
        neutra = NeuTraReparam(guide)
        model = poutine.infer_config(lambda _: {"reparam": neutra})
        model = poutine.reparam(model)
        nuts = NUTS(model)
        # ...now use the model in HMC or NUTS...

    Note that all sites must share a single common :class:`NeuTraReparam` instance.

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704

    :param ~pyro.infer.autoguide.AutoContinuous guide: A trained guide.
    """
    def __init__(self, guide):
        if not isinstance(guide, AutoContinuous):
            raise TypeError("NeuTraReparam expected an AutoContinuous guide, but got {}"
                            .format(type(guide)))
        self.guide
        self.x_constrained = []

    def __call__(self, name, fn, obs):
        log_density = 0.
        if not self.x_constrained:  # On first sample site.
            # Sample a shared latent.
            posterior = self.guide.get_posterior()
            if not isinstance(posterior, dist.TransformedDistribution):
                raise ValueError("NeuTraReparam only supports guides whose posteriors are "
                                 "TransformedDistributions but got a posterior of type {}"
                                 .format(type(posterior)))
            t = dist.transforms.ComposeTransform(posterior.transforms)
            z_unconstrained = pyro.sample("{}_latent".format(name),
                                          posterior.base_dist.mask(False))

            # Differentiably transform.
            x_unconstrained = t(z_unconstrained)
            log_density = t.log_abs_det_jacobian(z_unconstrained, x_unconstrained)
            x_constrained = self.guide._unpack_latent(x_unconstrained)
            self.x_constrained = list(reversed(x_constrained))

        # Extract a single site's value from the shared latent.
        site, value = self.x_unconstrained.pop()
        log_density = log_density + fn.log_prob(value)
        new_fn = dist.Delta(value, log_density, event_dim=fn.event_dim)
        return new_fn, value
