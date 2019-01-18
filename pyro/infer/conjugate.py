import pyro
from pyro.poutine.replay_messenger import ReplayMessenger


class SampleConjugateMessenger(ReplayMessenger):
    def _pyro_sample(self, msg):
        name = msg["name"]
        get_posterior = getattr(msg["fn"], "_posterior_latent_dist", None)
        get_leaf = getattr(msg["fn"], "_compounded_dist", None)
        if get_posterior and msg["is_observed"]:
            assert get_leaf is not None
            posterior_dist = get_posterior(msg["value"])
            latent_sample = pyro.sample(name + ".latent", posterior_dist)
            leaf_dist = get_leaf(latent_sample)
            msg["name"] = name + ".compounded"
            msg["fn"] = leaf_dist
        else:
            super(SampleConjugateMessenger, self)._pyro_sample(msg)
        return None


def infer_conjugate(fn=None, trace=None, params=None):
    msngr = SampleConjugateMessenger(trace, params)
    return msngr(fn) if fn is not None else msngr
