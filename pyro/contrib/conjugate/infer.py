import pyro.distributions as dist
from pyro.distributions.util import sum_leftmost
from pyro.poutine.messenger import Messenger
from pyro.poutine.replay_messenger import ReplayMessenger


def _make_cls(base, static_attrs, instance_attrs, parent_linkage=None):
    r"""
    Dynamically create classes named `_ + base.__name__`, which extend the
    base class with other optional instance and class attributes, and have
    a custom `.expand` method to propagate these attributes on expanded
    instances.

    :param cls base: Base class.
    :param dict static_attrs: static attributes to add to class.
    :param dict instance_attrs: instance attributes for initialization.
    :param str parent_linkage: attribute in the parent class that holds
        a reference to the distribution class.
    :return cls: dynamically generated class.
    """
    def _expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(cls, _instance)
        for attr in instance_attrs:
            setattr(new, attr, getattr(self, attr))
        if parent_linkage:
            setattr(new.parent, parent_linkage, new)
        return base.expand(self, batch_shape, _instance=new)

    name = "_" + base.__name__
    cls = type(name, (base,), instance_attrs)
    for k, v in static_attrs.items():
        setattr(cls, k, v)
    cls.expand = _expand
    return cls


def _latent(base, parent):
    return _make_cls(base, {"collapsible": True}, {"site_name": None, "parent": parent}, "_latent")


def _conditional(base, parent):
    return _make_cls(base, {"marginalize_latent": True}, {"parent": parent}, "_conditional")


def _compound(base, parent):
    return _make_cls(base, {}, {"parent": parent})


class BetaBinomialPair(object):
    def __init__(self):
        self._latent = None
        self._conditional = None

    def latent(self, *args, **kwargs):
        self._latent = _latent(dist.Beta, parent=self)(*args, **kwargs)
        return self._latent

    def conditional(self, *args, **kwargs):
        self._conditional = _conditional(dist.Binomial, parent=self)(*args, **kwargs)
        return self._conditional

    def posterior(self, obs):
        concentration1 = self._latent.concentration1
        concentration0 = self._latent.concentration0
        total_count = self._conditional.total_count
        reduce_dims = len(obs.size()) - len(concentration1.size())
        # Unexpand total_count to have the same shape as concentration0.
        # Raise exception if this isn't possible.
        total_count = sum_leftmost(total_count, reduce_dims)
        summed_obs = sum_leftmost(obs, reduce_dims)
        return dist.Beta(concentration1 + summed_obs,
                         total_count + concentration0 - summed_obs,
                         validate_args=self._latent._validate_args)

    def compound(self):
        return _compound(dist.BetaBinomial, parent=self)(concentration1=self._latent.concentration1,
                                                         concentration0=self._latent.concentration0,
                                                         total_count=self._conditional.total_count)


class GammaPoissonPair(object):
    def __init__(self):
        self._latent = None
        self._conditional = None

    def latent(self, *args, **kwargs):
        self._latent = _latent(dist.Gamma, parent=self)(*args, **kwargs)
        return self._latent

    def conditional(self, *args, **kwargs):
        self._conditional = _conditional(dist.Poisson, parent=self)(*args, **kwargs)
        return self._conditional

    def posterior(self, obs):
        concentration = self._latent.concentration
        rate = self._latent.rate
        reduce_dims = len(obs.size()) - len(rate.size())
        num_obs = obs.shape[:reduce_dims].numel()
        summed_obs = sum_leftmost(obs, reduce_dims)
        return dist.Gamma(concentration + summed_obs, rate + num_obs)

    def compound(self):
        return _compound(dist.GammaPoisson, parent=self)(concentration=self._latent.concentration,
                                                         rate=self._latent.rate)


class UncollapseConjugateMessenger(ReplayMessenger):
    r"""
    Extends `~pyro.poutine.replay_messenger.ReplayMessenger` to uncollapse
    compound distributions.
    """
    def _pyro_sample(self, msg):
        is_collapsible = getattr(msg["fn"], "collapsible", False)
        if is_collapsible:
            conj_node, parent = None, None
            for site_name in self.trace.observation_nodes + self.trace.stochastic_nodes:
                parent = getattr(self.trace.nodes[site_name]["fn"], "parent")
                if parent is not None and parent._latent.site_name == msg["name"]:
                    conj_node = self.trace.nodes[site_name]
                    break
            assert conj_node is not None, "Collapsible latent site `{}` with no corresponding conjugate site."\
                .format(msg["name"])
            msg["fn"] = parent.posterior(conj_node["value"])
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
    This replaces a latent-observed pair by collapsing the latent site
    (whose distribution has attribute `collapsible=True`), and replacing the
    observed site (whose distribution has attribute `marginalize_latent=True`)
    with a compound probability distribution that marginalizes out the latent
    site.
    """
    msngr = CollapseConjugateMessenger()
    return msngr(fn) if fn is not None else msngr
