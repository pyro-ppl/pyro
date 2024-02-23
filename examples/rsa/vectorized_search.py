import queue
from collections import OrderedDict
from typing import Tuple, Any, List, Dict

import torch

from pyro.infer.enum import iter_discrete_escape, iter_discrete_extend
from pyro.infer.abstract_infer import TracePosterior, TracePredictive
from pyro.poutine.trace_struct import Trace
from pyro.poutine.enum_messenger import EnumMessenger
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_site_shape
from pyro.distributions.util import is_validation_enabled
import pyro.poutine as poutine
import pyro.distributions as dist

from search_inference import memoize


class VectoredHashingMarginal(dist.Distribution):
    has_enumerate_support = True

    def __init__(self, trace_dist, sites=None) -> None:
        assert isinstance(trace_dist, TracePosterior), \
            "`trace_dist` must be trace posterior distribution object"
        
        self.sites = sites or "_RETURN"

        assert isinstance(self.sites, (str, list)), \
            "sites must be either '_RETURN' or list"

        if isinstance(self.sites, str):
            self.sites = [self.sites]

        super().__init__()
        self.trace_dist = trace_dist

    def _hash(self, v: Any) -> int:
        if torch.is_tensor(v):
            return hash(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            return hash(self._dict_to_tuple(v))
        else:
            return hash(torch.tensor(v))

    def _get_sample(self, nodes: OrderedDict, idx: int) -> float:
        samples = list(filter(lambda x: x["type"] == "sample", nodes.values()))
        if self.sites != ["_RETURN"]:
            samples = list(filter(lambda x: x["name"] in self.sites, samples))
        # print(samples)

        log_probs = [s["log_prob"] for s in samples]
        log_probs = [(log_prob[idx] if log_prob.ndim > 0 else log_prob) for log_prob in log_probs]
        return torch.Tensor(log_probs)

    @memoize(maxsize=100)
    def _dist_and_values(self) -> Tuple[dist.Distribution, OrderedDict]:
        # XXX currently this whole object is very inefficient
        vmap, logits = OrderedDict(), OrderedDict()
        for trace in self.trace_dist.exec_traces:
            # import pdb; pdb.set_trace()
            # TODO figure out how to convert vectorized sites into individual sites for storage
            _input = trace.nodes["_INPUT"]
            _return = trace.nodes["_RETURN"]

            for idx, val in enumerate(_return["value"]):
                logit = self._get_sample(trace.nodes, idx).sum()
                vhash = self._hash(val)
                # print(val, vhash)

                if vhash in logits:
                    logits[vhash] = dist.util.logsumexp(torch.stack([logits[vhash], logit], dim=-1))
                else:
                    logits[vhash] = logit
                    vmap[vhash] = _return["value"][idx]

        logits = torch.stack(list(logits.values())).contiguous().view(-1)
        logits = logits - dist.util.logsumexp(logits, dim=-1)
        d = dist.Categorical(logits=logits)
        return d, vmap

    def sample(self) -> float:
        d, vmap = self._dist_and_values()
        idx = d.sample()
        return list(vmap.values())[idx]

    def log_prob(self, val: Any) -> torch.Tensor:
        d, vmap = self._dist_and_values()
        vhash = self._hash(val)
        return d.log_prob(torch.tensor([list(vmap.keys()).index(vhash)]))

    def enumerate_support(self) -> List[float]:
        _, vmap = self._dist_and_values()
        return list(vmap.values())[:]

    def _dict_to_tuple(self, d: Dict) -> Tuple:
        if isinstance(d, dict):
            return tuple([(k, self._dict_to_tuple(d[k])) for k in sorted(d.keys())])
        return d

    def _weighted_mean(self, val, dim=0) -> float:
        weights = self._log_weights.reshape([-1, ] + (val.dim() - 1) * [1])
        max_weight = weights.max(dim=dim)[0]
        rel_probs  = (weights - max_weight).exp()
        return (val * rel_probs).sum(dim=dim) / rel_probs.sum(dim=dim)

    @property
    def mean(self) -> float:
        samples = torch.stack(list(self._dist_and_values()[1].values()))
        return self._weighted_mean(samples)

    @property
    def variance(self) -> float:
        samples = torch.stack(list(self._dist_and_values()[1].values()))
        stdev_squared = torch.pow(samples - self.mean, 2)
        return self._weighted_mean(stdev_squared)


class VectoredSearch(TracePosterior):
    def __init__(self, model, max_tries=int(1e6), **kwargs):
        self.model = model
        self.max_tries = max_tries
        self.max_plate_nesting = kwargs.pop("max_plate_nesting", 1)
        super().__init__(**kwargs)

    def _trace(self, model, args, kwargs) -> Tuple[Trace, torch.Tensor]:
        tr = poutine.trace(model, graph_type="flat").get_trace(*args, **kwargs)
        tr = prune_subsample_sites(tr)
        tr.compute_log_prob()
        # tr.compute_score_parts()
        # tr.pack_tensors()

        if is_validation_enabled():
            for site in tr.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, max_plate_nesting=self.max_plate_nesting)

        return tr, tr.log_prob_sum()

    def _traces(self, *args, **kwargs):
        # TODO implement a generator that yields results of vectorized enumeration
        q = queue.LifoQueue()
        q.put(poutine.Trace())

        model_enum = EnumMessenger(first_available_dim=-1 - self.max_plate_nesting)
        model = model_enum(self.model)
        model = poutine.queue(
            model,
            queue=q, 
            escape_fn=iter_discrete_escape,
            extend_fn=iter_discrete_extend
        )

        while not q.empty():
            yield self._trace(model, args, kwargs)
