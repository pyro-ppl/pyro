from __future__ import absolute_import, division, print_function

import numbers
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import torch
from six import add_metaclass

import pyro.poutine as poutine
from pyro.distributions import Categorical, Empirical
from pyro.ops.stats import waic


class EmpiricalMarginal(Empirical):
    """
    Marginal distribution over a single site (or multiple, provided they have the same
    shape) from the ``TracePosterior``'s model.

    ..note:: If multiple sites are specified, they must have the same tensor shape.
        Samples from each site will be stacked and stored within a single tensor. See
        :class:`~pyro.distributions.Empirical`. To hold the marginal distribution of sites
        having different shapes, use :class:`~pyro.infer.abstract_infer.Marginals` instead.

    :param TracePosterior trace_posterior: a ``TracePosterior`` instance representing
        a Monte Carlo posterior.
    :param list sites: optional list of sites for which we need to generate
        the marginal distribution.
    """

    def __init__(self, trace_posterior, sites=None, validate_args=None):
        assert isinstance(trace_posterior, TracePosterior), \
            "trace_dist must be trace posterior distribution object"
        if sites is None:
            sites = "_RETURN"
        self._num_chains = 1
        self._samples_buffer = defaultdict(list)
        self._weights_buffer = defaultdict(list)
        self._populate_traces(trace_posterior, sites)
        samples, weights = self._get_samples_and_weights()
        super(EmpiricalMarginal, self).__init__(samples,
                                                weights,
                                                validate_args=validate_args)

    def _get_samples_and_weights(self):
        """
        Appends values collected in the samples/weights buffers to their
        corresponding tensors.
        """
        num_chains = len(self._samples_buffer)
        samples_by_chain = []
        weights_by_chain = []
        for i in range(num_chains):
            samples_by_chain.append(torch.stack(self._samples_buffer[i], dim=0))
            weights_by_chain.append(torch.stack(self._weights_buffer[i], dim=0))
        if len(samples_by_chain) == 1:
            return samples_by_chain[0], weights_by_chain[0]
        else:
            return torch.stack(samples_by_chain, dim=0), torch.stack(weights_by_chain, dim=0)

    def _add_sample(self, value, log_weight=None, chain_id=0):
        """
        Adds a new data point to the sample. The values in successive calls to
        ``add`` must have the same tensor shape and size. Optionally, an
        importance weight can be specified via ``log_weight`` or ``weight``
        (default value of `1` is used if not specified).

        :param torch.Tensor value: tensor to add to the sample.
        :param torch.Tensor log_weight: log weight (optional) corresponding
            to the sample.
        :param int chain_id: chain id that generated the sample (optional).
            Note that if this argument is provided, ``chain_id`` must lie
            in ``[0, num_chains - 1]``, and there must be equal number
            of samples per chain.
        """
        weight_type = value.new_empty(1).float().type() if value.dtype in (torch.int32, torch.int64) \
            else value.type()
        # Apply default weight of 1.0.
        if log_weight is None:
            log_weight = torch.tensor(0.0).type(weight_type)
        if isinstance(log_weight, numbers.Number):
            log_weight = torch.tensor(log_weight).type(weight_type)
        if self._validate_args and log_weight.dim() > 0:
            raise ValueError("``weight.dim() > 0``, but weight should be a scalar.")

        # Append to the buffer list
        self._samples_buffer[chain_id].append(value)
        self._weights_buffer[chain_id].append(log_weight)
        self._num_chains = max(self._num_chains, chain_id + 1)

    def _populate_traces(self, trace_posterior, sites):
        assert isinstance(sites, (list, str))
        for tr, log_weight, chain_id in zip(trace_posterior.exec_traces,
                                            trace_posterior.log_weights,
                                            trace_posterior.chain_ids):
            value = tr.nodes[sites]["value"] if isinstance(sites, str) else \
                torch.stack([tr.nodes[site]["value"] for site in sites], 0)
            self._add_sample(value, log_weight=log_weight, chain_id=chain_id)


class Marginals(object):
    """
    Holds the marginal distribution over one or more sites from the ``TracePosterior``'s
    model. This is a convenience container class, which can be extended by ``TracePosterior``
    subclasses. e.g. for implementing diagnostics.

    :param TracePosterior trace_posterior: a TracePosterior instance representing
        a Monte Carlo posterior.
    :param list sites: optional list of sites for which we need to generate
        the marginal distribution.
    """
    def __init__(self, trace_posterior, sites=None, validate_args=None):
        assert isinstance(trace_posterior, TracePosterior), \
            "trace_dist must be trace posterior distribution object"
        if sites is None:
            sites = ["_RETURN"]
        elif isinstance(sites, str):
            sites = [sites]
        else:
            assert isinstance(sites, list)
        self.sites = sites
        self._marginals = OrderedDict()
        self._diagnostics = OrderedDict()
        self._trace_posterior = trace_posterior
        self._populate_traces(trace_posterior, validate_args)

    def _populate_traces(self, trace_posterior, validate):
        self._marginals = {site: EmpiricalMarginal(trace_posterior, site, validate)
                           for site in self.sites}

    def support(self, flatten=False):
        support = OrderedDict([(site, value.enumerate_support())
                               for site, value in self._marginals.items()])
        if self._trace_posterior.num_chains > 1 and flatten:
            for site, samples in support.items():
                shape = samples.size()
                flattened_shape = torch.Size((shape[0] * shape[1],)) + shape[2:]
                support[site] = samples.reshape(flattened_shape)
        return support

    @property
    def empirical(self):
        return self._marginals


@add_metaclass(ABCMeta)
class TracePosterior(object):
    """
    Abstract TracePosterior object from which posterior inference algorithms inherit.
    When run, collects a bag of execution traces from the approximate posterior.
    This is designed to be used by other utility classes like `EmpiricalMarginal`,
    that need access to the collected execution traces.
    """
    def __init__(self, num_chains=1):
        self.num_chains = num_chains
        self._reset()

    def _reset(self):
        self.log_weights = []
        self.exec_traces = []
        self.chain_ids = []  # chain id corresponding to the sample
        self._idx_by_chain = [[] for _ in range(self.num_chains)]  # indexes of samples by chain id
        self._categorical = None

    def marginal(self, sites=None):
        return Marginals(self, sites)

    @abstractmethod
    def _traces(self, *args, **kwargs):
        """
        Abstract method implemented by classes that inherit from `TracePosterior`.

        :return: Generator over ``(exec_trace, weight)`` or
        ``(exec_trace, weight, chain_id)``.
        """
        raise NotImplementedError("Inference algorithm must implement ``_traces``.")

    def __call__(self, *args, **kwargs):
        # To ensure deterministic sampling in the presence of multiple chains,
        # we get the index from ``idxs_by_chain`` instead of sampling from
        # the marginal directly.
        random_idx = self._categorical.sample().item()
        chain_idx, sample_idx = random_idx % self.num_chains, random_idx // self.num_chains
        sample_idx = self._idx_by_chain[chain_idx][sample_idx]
        trace = self.exec_traces[sample_idx].copy()
        for name in trace.observation_nodes:
            trace.remove_node(name)
        return trace

    def run(self, *args, **kwargs):
        """
        Calls `self._traces` to populate execution traces from a stochastic
        Pyro model.

        :param args: optional args taken by `self._traces`.
        :param kwargs: optional keywords args taken by `self._traces`.
        """
        self._reset()
        with poutine.block():
            for i, vals in enumerate(self._traces(*args, **kwargs)):
                if len(vals) == 2:
                    chain_id = 0
                    tr, logit = vals
                else:
                    tr, logit, chain_id = vals
                    assert chain_id < self.num_chains
                self.exec_traces.append(tr)
                self.log_weights.append(logit)
                self.chain_ids.append(chain_id)
                self._idx_by_chain[chain_id].append(i)
        self._categorical = Categorical(logits=torch.tensor(self.log_weights))
        return self

    def information_criterion(self, pointwise=False):
        """
        Computes information criterion of the model. Currently, returns only "Widely
        Applicable/Watanabe-Akaike Information Criterion" (WAIC) and the corresponding
        effective number of parameters.

        Reference:

        [1] `Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC`,
        Aki Vehtari, Andrew Gelman, and Jonah Gabry

        :param bool pointwise: a flag to decide if we want to get a vectorized WAIC or not. When
            ``pointwise=False``, returns the sum.
        :returns OrderedDict: a dictionary containing values of WAIC and its effective number of
            parameters.
        """
        if not self.exec_traces:
            return {}
        obs_node = None
        log_likelihoods = []
        for trace in self.exec_traces:
            obs_nodes = trace.observation_nodes
            if len(obs_nodes) > 1:
                raise ValueError("Infomation criterion calculation only works for models "
                                 "with one observation node.")
            if obs_node is None:
                obs_node = obs_nodes[0]
            elif obs_node != obs_nodes[0]:
                raise ValueError("Observation node has been changed, expected {} but got {}"
                                 .format(obs_node, obs_nodes[0]))

            log_likelihoods.append(trace.nodes[obs_node]["fn"]
                                   .log_prob(trace.nodes[obs_node]["value"]))

        ll = torch.stack(log_likelihoods, dim=0)
        waic_value, p_waic = waic(ll, ll.new_tensor(self.log_weights), pointwise)
        return OrderedDict([("waic", waic_value), ("p_waic", p_waic)])


class TracePredictive(TracePosterior):
    """
    Generates and holds traces from the posterior predictive distribution,
    given model execution traces from the approximate posterior. This is
    achieved by constraining latent sites to randomly sampled parameter
    values from the model execution traces and running the model forward
    to generate traces with new response ("_RETURN") sites.

    :param model: arbitrary Python callable containing Pyro primitives.
    :param TracePosterior posterior: trace posterior instance holding
        samples from the model's approximate posterior.
    :param int num_samples: number of samples to generate.
    """
    def __init__(self, model, posterior, num_samples):
        self.model = model
        self.posterior = posterior
        self.num_samples = num_samples
        super(TracePredictive, self).__init__()

    def _traces(self, *args, **kwargs):
        if not self.posterior.exec_traces:
            self.posterior.run(*args, **kwargs)
        for _ in range(self.num_samples):
            model_trace = self.posterior()
            replayed_trace = poutine.trace(poutine.replay(self.model, model_trace)).get_trace(*args, **kwargs)
            yield (replayed_trace, 0., 0)

    def marginal(self, sites=None):
        return self.posterior.marginal(sites)
