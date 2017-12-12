from __future__ import absolute_import, division, print_function

from pdb import set_trace as bb
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.mcmc.trace_kernel import TraceKernel
from pyro.util import ng_ones, ng_zeros
from torch import Tensor as T
from torch.autograd import Variable as V
from numpy.random import choice
from numpy import isfinite
from torch import log as tlog

_R = '_RETURN'


def VT(val):
    return V(T(val))


def VTA(val):
    return VT([val])


def assert_valid_trace(trace):
    assert not any(['fn' in n and isinstance(n['fn'], pyro._Subsample)
                    for nid, n in trace.nodes(data=True)]), \
                    "Single Site MH does not currently handle mapdata in model."


# get all the sample sites
def sample_sites(trace, minus=set()):
    return [nid for nid, n in trace.nodes(data=True)
            if nid not in minus and n["type"] == 'sample'
            and not n["is_observed"]]


class NormalProposal():
    def __init__(self, mu, sigma, tune_frequency=100):
        self.mu = mu
        self.sigma = sigma
        self.scale = 1.
        self.tune_frequency = tune_frequency
        self._tune_cnt = self.tune_frequency

    def sample(self, site_node):
        cur_val = site_node['value']
        mu = self.mu.expand_as(cur_val)
        sigma = self.scale*self.sigma

        # sigma just a number? expand to equal the must
        if tuple(sigma.data.shape) != mu.data.shape:
            sigma = sigma.expand_as(mu.data)

        return cur_val + dist.Normal(mu, sigma).sample()

    def log_pdf_given(self, y, given_x):
        # center our distribution at mu = given_x, then
        # get log pdf of y
        mu = given_x
        sigma = self.scale*self.sigma
        return dist.Normal(mu, sigma).log_pdf(y)

    # borrowed from:
    # https://github.com/mcleonard/sampyl/blob/master/sampyl/samplers/metropolis.py#L102
    # which in turn borrows from pymc3
    def tune(self, acceptance):
        self._tune_cnt -= 1
        if self._tune_cnt > 0:
            return self

        scale = self.scale
        # Switch statement
        if acceptance < 0.001:
            # reduce by 90 percent
            scale *= 0.1
        elif acceptance < 0.05:
            # reduce by 50 percent
            scale *= 0.5
        elif acceptance < 0.2:
            # reduce by ten percent
            scale *= 0.9
        elif acceptance > 0.95:
            # increase by factor of ten
            scale *= 10.0
        elif acceptance > 0.75:
            # increase by double
            scale *= 2.0
        elif acceptance > 0.5:
            # increase by ten percent
            scale *= 1.1

        self.scale = scale
        # reset our tuning count
        self._tune_cnt = self.tune_frequency
        return self


class MH(TraceKernel):
    def __init__(self, model, proposal_dist):
        self.model = model
        self.proposal_dist = proposal_dist

        # store all of our traces, and keep count on acceptance ratio
        self._kernel_traces = []
        self._accept_cnt = None
        self._tune_cnt = None
        self._call_cnt = None

        super(MH, self).__init__()

    def setup(self, *args, **kwargs):

        # haven't accepted any!
        self._accept_cnt = 0
        self._tune_cnt = 0
        self._call_cnt = 0

        # store our args call for later?
        self._args = args
        self._kwargs = kwargs

    def initial_trace(self):
        # maintain all traces, may need to kill at the end
        trace = poutine.trace(self.model).get_trace(*self._args, **self._kwargs)
        assert_valid_trace(trace)
        self._kernel_traces.append(trace)
        return trace

    def cleanup(self):
        for trace in self._kernel_traces:
            self._cleanup_trace(trace)
        self._kernel_traces = []

    def _cleanup_trace(self, reject_trace, accept_trace=None):
        pass

    @property
    def num_accepts(self):
        return self._accept_cnt

    def _get_trace_proposal(self, trace, r_site, proposal_value):

        # clone the only trace
        trace_proposal = trace.copy()

        # change in our trace_proposal
        trace_proposal.node[r_site]['value'] = proposal_value

        # where is this r_site selection?
        r_site_ix = list(trace_proposal.nodes()).index(r_site)

        # remove any sites post r_site
        removal_sites = [site_name
                         for site_ix, site_name in enumerate(trace_proposal.nodes())
                         if site_ix > r_site_ix]

        # remove all sites after r_site
        trace_proposal.remove_nodes_from(removal_sites)

        # all done, we have a new proposed trace, and there's nothing to replay
        # after the site.
        # TODO: More efficient removal of nodes for r_site (e.g. only dependencies)
        return trace_proposal

    def sample(self, trace):
        # sample called
        self._call_cnt += 1

        # 1. Random sample trace site to modify
        # 2. Get proposal for amount to modify site
        # 3. Adjust site
        # 4. Calculate prob for acceptance

        # how many nodes? == nodes - 1 because there is a _trace_uuid node
        trace_nodes = sample_sites(trace)
        trace_length = len(trace_nodes)

        # 1. randomly sample a site to modify in trace
        r_site = choice(trace_nodes)

        # 2. Get proposal for amount at modified site
        proposal_value = self.proposal_dist.sample(trace.node[r_site])
        trace_value = trace.node[r_site]['value']
        # get X and X'

        # get our trace to replay against
        replay_proposal = self._get_trace_proposal(trace, r_site, proposal_value)

        # replay against the trace proposal
        # depending on replay type, this can be pretty efficient!
        # TODO: Send r_site to know where to start replaying
        prop_trace = poutine.trace(poutine.replay(self.model, replay_proposal)).get_trace(*self._args, **self._kwargs)
        assert_valid_trace(prop_trace)

        # how many sample sites in our new trace
        prop_trace_nodes = sample_sites(prop_trace)
        prop_trace_length = len(prop_trace_nodes)

        # alg2 pg 772 http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf

        # get logp for both traces
        logp_original = trace.log_pdf()
        logp_proposal = prop_trace.log_pdf()

        # alg2 pg 772, line 7-8 http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf

        # R = -log(len(old_trace)) + PD(X').log_pdf(X)
        R = -tlog(VTA(trace_length)).type_as(trace_value) + \
            self.proposal_dist.log_pdf_given(trace_value, given_x=proposal_value)

        # F = -log(len(new_trace)) + PD(X).log_pdf(X')
        F = -tlog(VTA(prop_trace_length)).type_as(trace_value) + \
            self.proposal_dist.log_pdf_given(proposal_value, given_x=trace_value)

        # ll' - ll + R - F
        delta = (logp_proposal - logp_original) + (R - F)

        # get our delta y'all
        # alg2 line 12
        rand = pyro.sample('rand_t='.format(self._call_cnt),
                           dist.uniform, a=ng_zeros(1), b=ng_ones(1))
        accept_trace, reject_trace = None, None
        if isfinite(delta.data[0]) and rand.log().data[0] < delta.data[0]:
            # accept!
            self._accept_cnt += 1

            # accept the proposal, clean up the old trace
            accept_trace = prop_trace
            reject_trace = trace
        else:
            # keep the same trace, reject the proposed
            accept_trace = trace
            reject_trace = prop_trace

        # tune the proposal object -- if outside of tune_frequency, this will ignore
        self.proposal_dist.tune(self._accept_cnt/self._call_cnt)

        # handle accept/reject traces
        if accept_trace not in self._kernel_traces:
            # add the kernel
            self._kernel_traces.append(accept_trace)

        # then cleanup the accept, reject
        self._cleanup_trace(reject_trace, accept_trace=accept_trace)

        # tune the proposal object -- if outside of tune_frequency, this will ignore
        self.proposal_dist.tune(self.acceptance_ratio)

        # return old trace or new trace according to accepted or not
        return accept_trace

    @property
    def acceptance_ratio(self):
        return self._accept_cnt / self._call_cnt
