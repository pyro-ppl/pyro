import pyro
import pyro.distributions as dist


class TransformReparam:
    """
    Reparameterizer for
    :class:`pyro.distributions.torch.TransformedDistribution` .
    """
    def __call__(self, name, fn, obs):
        assert isinstance(fn, dist.TransformedDistribution)
        assert obs is None

        # Draw noise from the base distribution.
        y = pyro.sample("{}_base".format(name), fn.base_dist)

        # Differentiably transform.
        log_density = 0
        for t in fn.transforms:
            x, y = y, t(y)
            log_density = log_density + t.log_abs_det_jacobian(x, y)

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(x, log_density, event_dim=fn.event_dim)
        return new_fn, y
