
class TransformReparam:
    """
    Arbitrary transform reparameterizer.

    This can be used to reparameterize wrt an arbitrary bijective
    :class:`~torch.distributions.transforms.Transform` object, and requires:

    - for sample sites, the forward ``.__call__()`` method and the
      ``.log_abs_det_jacobian()`` transform to be defined, as in [1]
    - for observation sites, requires the backward ``.inv.__call__()`` method
      and the ``.log_abs_det_jacobian()``.

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704

    :param ~pyro.distributions.TorchDistribution base_dist: A base
        distribution for the auxiliary latent variable.
    :param ~torch.distributions.transforms.Transform transform: A bijective
        transform defining forward and log abs det jacobian methods.
    """
    def __init__(self, base_dist, transform):
        self.base_dist = base_dist
        self.transform = transform

    def __call__(self, name, fn, obs):
        # FIXME this does not play well with poutine.condition and poutine.replay
        if obs is None:
            z = pyro.sample("{}_base".format(name), self.base_dist)
            x = self.transform(z)
        else:
            x = obs
            z = self.transform.inv(x)
            pyro.sample("{}_base".format(name), self.base_dist, obs=z)

        # Simulate a pyro.deterministic() site.
        log_density = (fn.log_prob(x)
                       - self.base_dist.log_prob(z)
                       + self.transform.log_abs_det_jacobian(z, x))
        new_fn = dist.Delta(x, log_density, event_dim=fn.event_dim)
        return new_fn, x
