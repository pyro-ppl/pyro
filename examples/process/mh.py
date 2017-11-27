from __future__ import absolute_import, division, print_function

from pdb import set_trace as bb
import pyro
import pyro.distributions as dist
# from pyro.infer.mcmc.trace_kernel import TraceKernel
from mcmc.trace_kernel import TraceKernel
from pyro.util import ng_ones, ng_zeros
from command_points import ForkContinueCommand, NudgeForkContinueCommand
from torch import Tensor as T
from torch.autograd import Variable as V
from fork_poutine import ForkPoutine, _R
from numpy.random import choice
from numpy import isnan, isinf, isfinite
from torch import log as tlog


def VT(val):
    return V(T(val))


def VTA(val):
    return VT([val])


# get all the sample sites
def sample_sites(trace, minus=set()):
    return [nid for nid, n in trace.nodes(data=True)
            if nid not in minus and n["type"] == 'sample']


class NormalProposal():
    def __init__(self, mu, sigma, tune_frequency=100):
        self.mu = mu
        self.sigma = sigma
        self.scale = 1.
        self.tune_frequency = tune_frequency
        self._tune_cnt = self.tune_frequency

    def sample(self, mu_size):
        return self(self.mu.expand(mu_size)).sample()

    def __call__(self, mu):
        sigma = self.scale*self.sigma
        # sigma just a number? expand to equal the must
        if tuple(sigma.data.shape) != mu.data.shape:
            sigma = sigma.expand_as(mu.data)
        return dist.Normal(mu, sigma)

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
        self.trace_keys = None
        self._is_initialized = False
        self._dist_traces = []
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

        # use our poutine to get command access to the model
        print("INITIALIZING FORKS")
        self.fork_control = ForkPoutine(self.model, ForkContinueCommand).initialize(1)

        # get a singular trace -- synchronous call here, running trace on another thread
        self._prototype_trace = self.fork_control.get_trace(*args, **kwargs)

        # check on valid trace
        prototype_trace_log_pdf = self._prototype_trace.log_pdf().data[0]
        if isnan(prototype_trace_log_pdf) or isinf(prototype_trace_log_pdf):
            raise ValueError('Model specification incorrect - trace log pdf is NaN, Inf or 0.')

        # setup returns the first trace to run
        return self._prototype_trace

    def cleanup(self):
        print("CLEANING UP POST-MH")
        bb()

        # sync kill all forks and threads
        self.fork_control.kill_all()

    @property
    def num_accepts(self):
        return self._accept_cnt

    def sample(self, trace, time_step):

        ret_trace = trace
        self._call_cnt += 1

        # 1. Random sample trace site to modify
        # 2. Get proposal for amount to modify site
        # 3. Sync call continue with nudge from modified state
        # 4. Calculate values for delta choice
        # how many nodes? == nodes - 1 because there is a _trace_uuid node
        trace_nodes = sample_sites(trace)
        trace_length = len(trace_nodes)

        # 1. randomly sample a site to modify in trace
        r_site = choice(trace_nodes)

        # now we create a nudge amount for each var in the sample
        nudge_amt = self.proposal_dist.sample(trace.node[r_site]['value'].data.shape)

        # now nudge and continue
        prop_trace = self.fork_control.continue_trace(trace, r_site,
                                                      NudgeForkContinueCommand(nudge_amt,
                                                                               preserve_parent=True))

        prop_trace_nodes = sample_sites(prop_trace)
        prop_trace_length = len(prop_trace_nodes)

        # alg2 pg 772 http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf

        # get logp for both traces
        logp_original = trace.node[_R()]["batch_log_pdf"].view(-1)
        logp_proposal = prop_trace.node[_R()]["batch_log_pdf"].view(-1)

        # get our sample at the proposed site
        # get X and X'
        x_original = trace.node[r_site]["value"]
        x_proposal = prop_trace.node[r_site]["value"]

        # alg2 pg 772, line 7-8 http://proceedings.mlr.press/v15/wingate11a/wingate11a.pdf
        # R = -log(len(old_trace)) + PD(X').log_pdf(X)
        # F = -log(len(new_trace)) + PD(X).log_pdf(X')
        R = -tlog(VTA(trace_length)).type_as(x_original) + \
            self.proposal_dist(x_proposal).log_pdf(x_original)

        F = -tlog(VTA(prop_trace_length)).type_as(x_original) + \
            self.proposal_dist(x_original).log_pdf(x_proposal)

        # ll' - ll + R - F
        delta = (logp_proposal - logp_original + R - F)

        # get our delta y'all
        # alg2 line 12
        rand = pyro.sample('rand_t='.format(time_step), dist.uniform, a=ng_zeros(1), b=ng_ones(1))
        if isfinite(delta.data[0]) and rand.log().data[0] < delta.data[0]:
            self._accept_cnt += 1
            ret_trace = prop_trace
            # we can kill the old trace, nothing desirable here
            self.fork_control.kill_trace(trace, r_site)
        else:
            # kill the whole proposed trace otherwise -- it was rejected!
            self.fork_control.kill_trace(prop_trace, "_INPUT")

        # tune the proposal object -- if outside of tune_frequency, this will ignore
        self.proposal_dist.tune(self._accept_cnt/self._call_cnt)

        # return old trace or new trace according to accepted or not
        return ret_trace
