from __future__ import absolute_import, division, print_function

import logging

import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.discrete import infer_discrete
from pyro.optim import Adam


class Mixture(object):
    """
    A mixture model of sparse heterogeneous tabular data.

    :param list features: A list of
        :class:`~pyro.contrib.tabular.features.Feature` objects defining a
        feature model for each column. Feature models can be repeated,
        indicating that two columns share a common feature model (with shared
        learned parameters). Features should reside on the same device as data.
    :param int capacity: Tne number of mixture components.
    """
    def __init__(self, features, capacity=64):
        assert capacity > 1
        self.features = features
        self.capacity = capacity
        self.guide = AutoDelta(poutine.block(self.model, hide=["mixture_z"]))

    def __getstate__(self):
        return {"features": self.features, "capacity": self.capacity}

    def __setstate__(self, state):
        self.__init__(**state)

    def model(self, data, num_rows=None, impute=False):
        """
        :param list data: batch of heterogeneous column-oriented data.  Each
            column should be either a torch.Tensor (if observed) or None (if
            unobserved).
        :param int num_rows: Optional number of rows in entire dataset, if data
            is is a minibatch.
        :param bool impute: Whether to impute missing features. This should be
            set to False during training and True when making predictions.
        :returns: a copy of the input data, optionally with missing columns
            stochastically imputed according the joint posterior.
        :rtype: list
        """
        assert len(data) == len(self.features)
        assert not all(column is None for column in data)
        device = next(col.device for col in data if col is not None)
        batch_size = next(col.size(0) for col in data if col is not None)
        if num_rows is None:
            num_rows = batch_size

        # Sample a mixture model for each feature.
        mixtures = [None] * len(self.features)
        components_plate = pyro.plate("components_plate", self.capacity, dim=-1)
        for v, feature in enumerate(self.features):
            shared = feature.sample_shared()
            with components_plate:
                mixtures[v] = feature.sample_group(shared)

        # Sample latent class distribution from a Dirichlet prior.
        concentration = torch.full((self.capacity,), 0.5, device=device)
        probs = pyro.sample("mixture_probs", dist.Dirichlet(concentration))

        # Sample data-local variables.
        subsample = None if (batch_size == num_rows) else [None] * batch_size
        with pyro.plate("data", num_rows, subsample=subsample, dim=-1):

            # Sample latent class.
            z = pyro.sample("mixture_z", dist.Categorical(probs),
                            infer={"enumerate": "parallel"})

            # Sample observed features conditioned on latent classes.
            x = [None] * len(self.features)
            for v, feature in enumerate(self.features):
                if data[v] is not None or impute:
                    x[v] = pyro.sample("mixture_x_{}".format(v),
                                       feature.value_dist(mixtures[v], component=z),
                                       obs=data[v])

        return x

    def trainer(self, optim=None):
        return MixtureTrainer(self, optim)

    def sample(self, data, num_samples=None):
        """
        Sample missing data conditioned on observed data.
        """
        model = self.model
        guide = self.guide
        first_available_dim = -2

        # Optionally draw vectorized samples.
        if num_samples is not None:
            plate = pyro.plate("num_samples_vectorized", num_samples,
                               dim=first_available_dim)
            model = plate(model)
            guide = plate(guide)
            first_available_dim -= 1

        # Sample global parameters from the guide.
        guide_trace = poutine.trace(guide).get_trace(data)
        model = poutine.replay(model, guide_trace)

        # Sample local latent variables using variable elimination.
        model = infer_discrete(model, first_available_dim=first_available_dim)
        return model(data, impute=True)

    def log_prob(self, data):
        """
        Compute posterior preditive probability of partially observed data.
        """
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        loss = elbo.differentiable_loss if torch.is_grad_enabled() else elbo.loss
        return loss(self.model, self.guide, data)


class MixtureTrainer(object):
    """
    Initializes and trains a :class:`Mixture` model.

    :param Mixture model: A Mixture model to train.
    :param pyro.optim.optim.PyroOptim optim: A Pyro optimizer to learn feature
        parameters.
    """
    def __init__(self, model, optim=None):
        assert isinstance(model, Mixture)
        if optim is None:
            optim = Adam({})
        self._elbo = TraceEnum_ELBO(max_plate_nesting=1)
        self._svi = SVI(model.model, model.guide, optim, self._elbo)
        self._model = model

    def init(self, data, init_groups=True):
        assert len(data) == len(self._model.features)
        for feature, column in zip(self._model.features, data):
            if column is not None:
                feature.init(column)
        if init_groups:
            self._elbo.loss(self._model.model, self._model.guide, data)

    def step(self, data, num_rows=None):
        loss = self._svi.step(data, num_rows=num_rows)

        # Log metrics to diagnose convergence issues.
        if logging.Logger(None).isEnabledFor(logging.DEBUG):
            with poutine.block():
                probs = pyro.param("auto_mixture_probs")
            entropy = -(probs * probs.log()).sum()
            perplexity = entropy.exp()
            logging.debug("perplexity: {:.1f}".format(perplexity))

        return loss
