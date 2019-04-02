from functools import reduce
from operator import mul

import pyro.distributions as dist
from pyro.distributions.util import sum_leftmost
from pyro.poutine.messenger import Messenger
from pyro.poutine.replay_messenger import ReplayMessenger


class _Beta(dist.Beta):
    collapsible = True

    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        self.site_name = None
        super(_Beta, self).__init__(*args, **kwargs)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_Beta, _instance)
        new.site_name = self.site_name
        new.parent = self.parent
        new.parent._latent = new
        return super(_Beta, self).expand(batch_shape, _instance=new)


class _Binomial(dist.Binomial):
    marginalize_latent = True

    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        super(_Binomial, self).__init__(*args, **kwargs)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_Binomial, _instance)
        new.parent = self.parent
        new.parent._conditional = new
        return super(_Binomial, self).expand(batch_shape, _instance=new)


class _BetaBinomial(dist.BetaBinomial):
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent
        super(_BetaBinomial, self).__init__(*args, **kwargs)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_BetaBinomial, _instance)
        new.parent = self.parent
        self.parent._conditional = self
        return super(_BetaBinomial, self).expand(batch_shape, _instance=new)


class BetaBinomialPair(object):
    def __init__(self):
        self._latent = None
        self._conditional = None

    def latent(self, *args, **kwargs):
        self._latent = _Beta(self, *args, **kwargs)
        return self._latent

    def conditional(self, *args, **kwargs):
        self._conditional = _Binomial(self, *args, **kwargs)
        return self._conditional

    def posterior(self, obs):
        concentration1 = self._latent.concentration1
        concentration0 = self._latent.concentration0
        total_count = self._conditional.total_count
        reduce_dims = len(obs.size()) - len(concentration1.size())
        num_obs = reduce(mul, obs.size()[:reduce_dims], 1)
        total_count = total_count[tuple(slice(1) if i < reduce_dims else slice(None)
                                  for i in range(total_count.dim()))].reshape(concentration0.shape)
        summed_obs = sum_leftmost(obs, reduce_dims)
        return dist.Beta(concentration1 + summed_obs,
                         num_obs * total_count + concentration0 - summed_obs,
                         validate_args=self._latent._validate_args)

    def compound(self):
        return _BetaBinomial(self,
                             concentration1=self._latent.concentration1,
                             concentration0=self._latent.concentration0,
                             total_count=self._conditional.total_count)


class UncollapseConjugateMessenger(ReplayMessenger):
    r"""
    Extends `~pyro.poutine.replay_messenger.ReplayMessenger` to uncollapse
    compound distributions. Note that if the original collapsed observed site
    was named "x", it is replaced with a sample site named "x.latent" followed
    by an observed site name "x.predictive".
    """
    def _pyro_sample(self, msg):
        is_collapsible = getattr(msg["fn"], "collapsible", False)
        if is_collapsible:
            observed_node, conj_pair = None, None
            for site_name in self.trace.observation_nodes:
                conj_pair = getattr(self.trace.nodes[site_name]["fn"], "parent")
                if conj_pair is not None and conj_pair._latent.site_name == msg["name"]:
                    observed_node = self.trace.nodes[site_name]
                    break
            assert observed_node is not None, "Collapsible latent site `{}` with no observed."\
                .format(msg["name"])
            msg["fn"] = conj_pair.posterior(observed_node["value"])
            msg["value"] = msg["fn"].sample()
        else:
            return super(UncollapseConjugateMessenger, self)._pyro_sample(msg)


def uncollapse_conjugate(fn=None, trace=None, params=None):
    r"""
    This extends the behavior of :function:`~pyro.poutine.replay` poutine, so that in
    addition to replaying the values at sample sites from the ``trace`` in the
    original callable ``fn`` when the same sites are sampled, this also "uncollapses"
    any observed compound distributions (defined in :module:`pyro.distributions.conjugate`)
    by sampling the originally collapsed parameter values from its posterior distribution
    followed by observing the data with the sampled parameter values.
    """
    msngr = UncollapseConjugateMessenger(trace, params)
    return msngr(fn) if fn is not None else msngr


class CollapseConjugateMessenger(Messenger):
    def _pyro_sample(self, msg):
        is_collapsible = getattr(msg["fn"], "collapsible", False)
        marginalize_latent = getattr(msg["fn"], "marginalize_latent", False)
        if is_collapsible:
            msg["fn"].site_name = msg["name"]
            msg["stop"] = True
        elif marginalize_latent:
            msg["fn"] = msg["fn"].parent.compound()
        else:
            return


def collapse_conjugate(fn=None):
    r"""
    This simply collapses (removes from the trace) and sample sites that have
    `collapse=True` set in their `infer` config.
    """
    msngr = CollapseConjugateMessenger()
    return msngr(fn) if fn is not None else msngr
