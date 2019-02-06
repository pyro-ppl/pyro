from pyro.poutine.messenger import Messenger
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
        get_predictive = getattr(msg["fn"], "_posterior_predictive", None)
        if dict.get(msg["infer"], "collapsed", False):
            observed = None
            parent_name = name.split(".")[0]
            for name in self.trace.observation_nodes:
                if self.trace.nodes[name]["fn"].name == parent_name:
                    observed = self.trace.nodes[name]["value"]
                    parent_dist = self.trace.nodes[name]["fn"]
                    break
            if observed is None:
                raise RuntimeError("Collapsed latent site without corresponding observe.")
            msg["fn"] = parent_dist._posterior_latent(observed)
            sample = msg["fn"].sample()
            parent_dist._latent_sample = msg["value"] = sample
        elif get_predictive and msg["is_observed"]:
            latent_sample = self.trace.nodes[name]["fn"]._latent_sample
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


class CollapseConjugateMessenger(Messenger):
    def _pyro_sample(self, msg):
        infer = msg["infer"]
        if infer.get("collapse", False):
            msg["stop"] = True


def collapse_conjugate(fn=None):
    r"""
    """
    msngr = CollapseConjugateMessenger()
    return msngr(fn) if fn is not None else msngr
