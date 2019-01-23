import pyro
from pyro.poutine.replay_messenger import ReplayMessenger


class SampleConjugateMessenger(ReplayMessenger):
    r"""
    Extends `~pyro.poutine.replay_messenger.ReplayMessenger` to uncollapse
    compound distributions. Note that if the original collapsed observed site
    was named "x", it is replaced with a sample site named "x.latent" followed
    by an observed site name "x.predictive".
    """
    def _pyro_sample(self, msg):
        name = msg["name"]
        get_posterior = getattr(msg["fn"], "_posterior_latent", None)
        get_predictive = getattr(msg["fn"], "_posterior_predictive", None)
        if get_posterior and msg["is_observed"]:
            assert get_predictive is not None
            posterior_dist = get_posterior(msg["value"])
            latent_sample = pyro.sample(name + ".latent", posterior_dist)
            leaf_dist = get_predictive(latent_sample)
            msg["name"] = name + ".predictive"
            msg["fn"] = leaf_dist
        else:
            super(SampleConjugateMessenger, self)._pyro_sample(msg)
        return None


def infer_conjugate(fn=None, trace=None, params=None):
    r"""
    This extends the behavior of :function:`~pyro.poutine.replay` poutine, so that in
    addition to replaying the values at sample sites from the ``trace`` in the
    original callable ``fn`` when the same sites are sampled, this also "uncollapses"
    any observed compound distributions (defined in :module:`pyro.distributions.conjugate`)
    by sampling the originally collapsed parameter values from its posterior distribution
    followed by observing the data with the sampled parameter values.
    """
    msngr = SampleConjugateMessenger(trace, params)
    return msngr(fn) if fn is not None else msngr
