# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro.distributions as dist

from .reparam import Reparam


class LinearHMMReparam(Reparam):
    """
    Auxiliary variable reparameterizer for
    :class:`~pyro.distributions.LinearHMM` random variables.

    This defers to component reparameterizers to create auxiliary random
    variables conditioned on which the process becomes a
    :class:`~pyro.distributions.GaussianHMM` . If the ``observation_dist`` is a
    :class:`~pyro.distributions.TransformedDistribution` this reorders those
    transforms so that the result is a
    :class:`~pyro.distributions.TransformedDistribution` of
    :class:`~pyro.distributions.GaussianHMM` .

    This is useful for training the parameters of a
    :class:`~pyro.distributions.LinearHMM` distribution, whose
    :meth:`~pyro.distributions.LinearHMM.log_prob` method is undefined.  To
    perform inference in the presence of non-Gaussian factors such as
    :meth:`~pyro.distributions.Stable`, :meth:`~pyro.distributions.StudentT` or
    :meth:`~pyro.distributions.LogNormal` , configure with
    :class:`~pyro.infer.reparam.studentt.StudentTReparam` ,
    :class:`~pyro.infer.reparam.stable.StableReparam` ,
    :class:`~pyro.infer.reparam.stable.SymmetricStableReparam` , etc.  component
    reparameterizers for ``init``, ``trans``, and ``scale``. For example::

        hmm = LinearHMM(
            init_dist=Stable(1,0,1,0).expand([2]).to_event(1),
            trans_matrix=torch.eye(2),
            trans_dist=MultivariateNormal(torch.zeros(2), torch.eye(2)),
            obs_matrix=torch.eye(2),
            obs_dist=TransformedDistribution(
                Stable(1.5,-0.5,1.0).expand([2]).to_event(1),
                ExpTransform()))

        rep = LinearHMMReparam(init=SymmetricStableReparam(),
                               obs=StableReparam())

        with poutine.reparam(config={"hmm": rep}):
            pyro.sample("hmm", hmm, obs=data)

    :param init: Optional reparameterizer for the initial distribution.
    :type init: ~pyro.infer.reparam.reparam.Reparam
    :param trans: Optional reparameterizer for the transition distribution.
    :type trans: ~pyro.infer.reparam.reparam.Reparam
    :param obs: Optional reparameterizer for the observation distribution.
    :type obs: ~pyro.infer.reparam.reparam.Reparam
    """
    def __init__(self, init=None, trans=None, obs=None):
        assert init is None or isinstance(init, Reparam)
        assert trans is None or isinstance(trans, Reparam)
        assert obs is None or isinstance(obs, Reparam)
        self.init = init
        self.trans = trans
        self.obs = obs

    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, (dist.LinearHMM, dist.IndependentHMM))
        if fn.duration is None:
            raise ValueError("LinearHMMReparam requires duration to be specified "
                             "on targeted LinearHMM distributions")

        # Unwrap IndependentHMM.
        if isinstance(fn, dist.IndependentHMM):
            if obs is not None:
                obs = obs.transpose(-1, -2).unsqueeze(-1)
            hmm, obs = self(name, fn.base_dist.to_event(1), obs)
            hmm = dist.IndependentHMM(hmm.to_event(-1))
            if obs is not None:
                obs = obs.squeeze(-1).transpose(-1, -2)
            return hmm, obs

        # Reparameterize the initial distribution as conditionally Gaussian.
        init_dist = fn.initial_dist
        if self.init is not None:
            init_dist, _ = self.init("{}_init".format(name),
                                     self._wrap(init_dist, event_dim - 1), None)
            init_dist = init_dist.to_event(1 - init_dist.event_dim)

        # Reparameterize the transition distribution as conditionally Gaussian.
        trans_dist = fn.transition_dist
        if self.trans is not None:
            if trans_dist.batch_shape[-1] != fn.duration:
                trans_dist = trans_dist.expand(trans_dist.batch_shape[:-1] + (fn.duration,))
            trans_dist, _ = self.trans("{}_trans".format(name),
                                       self._wrap(trans_dist, event_dim), None)
            trans_dist = trans_dist.to_event(1 - trans_dist.event_dim)

        # Reparameterize the observation distribution as conditionally Gaussian.
        obs_dist = fn.observation_dist
        if self.obs is not None:
            if obs_dist.batch_shape[-1] != fn.duration:
                obs_dist = obs_dist.expand(obs_dist.batch_shape[:-1] + (fn.duration,))
            obs_dist, obs = self.obs("{}_obs".format(name),
                                     self._wrap(obs_dist, event_dim), obs)
            obs_dist = obs_dist.to_event(1 - obs_dist.event_dim)

        # Reparameterize the entire HMM as conditionally Gaussian.
        hmm = dist.GaussianHMM(init_dist, fn.transition_matrix, trans_dist,
                               fn.observation_matrix, obs_dist, duration=fn.duration)
        hmm = self._wrap(hmm, event_dim)

        # Apply any observation transforms.
        if fn.transforms:
            hmm = dist.TransformedDistribution(hmm, fn.transforms)

        return hmm, obs
