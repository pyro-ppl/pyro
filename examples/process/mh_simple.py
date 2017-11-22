from pdb import set_trace as bb
import multiprocessing as mp
import numpy as np
from multiprocessing import Process
from numpy.random import choice
import time
import pyro
from networkx import DiGraph
from pyro.poutine import TracePoutine
import pyro.distributions as dist
from pickle import loads
from torch import Tensor as T
from torch.autograd import Variable as V
from command_points import add_control_point, ForkContinueCommand, LogPdfCommand
from command_points import ResampleForkContinueCommand, ResampleCloneContinueCommand, CloneCommand, KillCommand
from ip_communication import get_uuid, multi_try, RTraces, RLock, RMessages, RPairs
from collections import defaultdict


def VT(val):
    return V(T(val))


def VTA(val):
    return VT([val])


# simple fork just takes a command and applies it at every site encountered
class ForkPoutine(TracePoutine):
    def __init__(self, fn, CommandCls):

        # each trace is assigned a unique ID by default
        self.trace_uuid = get_uuid()
        self.CommandCls = CommandCls
        # same as normal trace poutine
        super(ForkPoutine, self).__init__(fn)

    # override the sample, we'll add a control point
    def _pyro_sample(self, msg):
        # when asked to sample, instead we add a control point with trace
        site_name = msg["name"]

        # we've already seen this sample point, and we are calling for a resample
        # remove the node
        if "is_forked" in msg:
            # kill and replace point. don't add another control point, that's redundant
            self.trace.remove_node(site_name)
            msg["done"] = False
            rval = super(ForkPoutine, self)._pyro_sample(msg)
            return rval
        else:
            # mark as forked for later calls to sample (e.g. resampling at a site)
            msg["is_forked"] = True

            # sampel like normal first, then we'll add a control point to later adjust what happened
            super(ForkPoutine, self)._pyro_sample(msg)
            # Poutine._pyro_sample(self, msg)

            # then we're going to run our command control
            # currently passing self to modify the trace_uuid only
            # TODO: Don't pass self, do this better
            add_control_point(self.trace_uuid, site_name, self, self.CommandCls(), msg)

            print("Post control point sample msg: {}".format(msg))

            # get the value of the node, whatever it's set to now
            return self.trace.nodes(data='value')[site_name]

    def __call__(self, *args, **kwargs):
        """
        Adds appropriate edges based on cond_indep_stack information
        upon exiting the context.
        """
        # get our normal __call__ setup
        r_val = super(ForkPoutine, self).__call__(*args, **kwargs)

        msg = {
            "type": "return",
            "args": args,
            "kwargs": kwargs
        }

        # append batch_log_pdf to our _RETURN node before serializing in control points
        # this is highly inappropriate
        # TODO: adjust pyro.poutine.Trace to allow appending to a trace node with a force command
        DiGraph.add_node(self.trace, "_RETURN", batch_log_pdf=self.trace.batch_log_pdf())

        # create a final control point that we can use to calculate other values
        add_control_point(self.trace_uuid, "_RETURN", self, self.CommandCls(), msg)

        # all done
        return r_val


class MetropolisHastings():

    def __init__(self, model, num_particles):
        self.num_particles = num_particles
        self.model = model
        self.traces = RTraces()
        self.locks = RLock()
        self.fork_pairs = RPairs()
        self.log_msgs = RMessages()
        self.trace_keys = None
        self._is_initialized = False
        self._dist_traces = []

    def __call__(self, *args, **kwargs):
        self.step(*args, **kwargs)

    def _initialize(self):
        # create a bunch of traces to begin with
        self.all_processes = [try_start_process(target=run_fork_model, args=[])
                              for i in range(self.num_particles)]

        # wait for all of our finished keys
        finished_keys = self.sync_get_return_keys(self.num_particles)

        # store all relevant traces
        self.trace_keys = set(map(RTraces.get_trace_from_key, finished_keys))

        # now we have a collection of finished traces, we can proceed like normal
        self._is_initialized = True

    def sync_get_db_return(self, total_keys, rdb_call, match="*_RETURN", *args, **kwargs):

        # now all of the processes are running, we wait until we have completed particles
        finished_vals = rdb_call(match)

        # loop forever
        # TODO: Add timeout
        while len(finished_vals) < total_keys:
            finished_vals = rdb_call(match)

        # all done, send back keys for all of the finished traces
        return finished_vals

    def save_trace(self, trace_uuid, trace_obj):
        self._dist_traces.append((trace_uuid, trace_obj))

    def step(self, *args, **kwargs):

        if not self._is_initialized:
            self._initialize()

        # now we need to go through our particles
        # 1. get all particles and sites
        # 2. randomly select site for each particle
        # 3. resample and continue till end
        # 4. collect the log pdfs of the new traces
        # 5. accept/reject the different particles
        # 6. kill the old forks
        # 7. get our active keys

        # 1. get all of the information about our trace+site
        all_site_uuids = [tid for tid in self.traces.get_all_keys() if '_RETURN' not in tid]

        # now we group by type
        trace_to_sites = defaultdict(list)
        for site_uuid in all_site_uuids:
            trace_uuid = RTraces.get_trace_from_key(site_uuid)
            trace_to_sites[trace_uuid].append(site_uuid)

        # now we have a list of all of our particles
        # 2. here we simply choose random sites to manipulate
        site_samples = {trace_uuid: choice(sites)
                        for trace_uuid, sites in trace_to_sites.items()}

        # release the locks and resample fork
        for lock_uuid in site_samples.values():
            # Preserve the original site in case we reject and want to branch from here again
            # TODO: logic for MH says we should ignore all remaining forks and just store our _RETURN
            # TODO: This is a generic resample (orig to test logic), but actually we need to nudge with proposal
            self.locks.release_lock(lock_uuid, ResampleForkContinueCommand(seed=None, preserve_parent=True))

        # should have this many _RETURN traces
        post_lock_size = len(self.trace_keys) + len(site_samples)

        # 3. go until we get all of our returns
        all_trace_keys = self.sync_get_db_return(post_lock_size,
                                                 rdb_call=self.traces.get_all_keys,
                                                 match="*_RETURN")

        # we know all the keys are there, get all our finished traces
        finished_traces = {RTraces.get_trace_from_key(site_uuid): trace_obj
                           for site_uuid, trace_obj in self.traces.get_all_items(match="*_RETURN")}

        # 4. get all the log pdfs
        trace_to_log_pdf = {trace_uuid: trace.nodes(data='batch_log_pdf')[RTraces.get_trace_key(trace_uuid, "_RETURN")]
                            for trace_uuid, trace in finished_traces.items()}

        # Get the resampled mappings
        prev_to_new_trace_map = dict(map(RPairs.get_pair_from_key, self.fork_pairs.get_all_keys()))

        kill_traces = []

        # by assumption, we made one choice per
        for orig_trace_uuid in site_samples:
            assert orig_trace_uuid in prev_to_new_trace_map, "Unknown mapping, did you miss a lock?"
            new_trace_uuid = prev_to_new_trace_map[orig_trace_uuid]

            # comapre log of orig vs log of proposed
            logp_orig = trace_to_log_pdf[orig_trace_uuid]
            logp_proposal = trace_to_log_pdf[new_trace_uuid]

            delta = (logp_proposal - logp_orig).data[0]
            # 5. accept according to ratio logp_proposed/logp_orig
            # https://github.com/mcleonard/sampyl/blob/master/sampyl/samplers/metropolis.py#L95
            if np.isfinite(delta) and np.log(np.random.uniform()) < delta:
                # accept!
                #
                print("add the trace internally (if we're post burn-in)")
                self.save_trace(new_trace_uuid, finished_traces[new_trace_uuid])

            else:
                # reject!
                # kill any command points, traces, locks, or anything associated with this
                # issue kill commands to all sites
                kill_traces.append(new_trace_uuid)

        kill_traces = set(kill_traces)
        kill_sites = [site_uuid for site_uuid in all_trace_keys if RTraces.get_trace_from_key(site_uuid) in kill_traces]

        # 6. kill off all the rejected proposals. we don't need any of those forks or children
        for lock_uuid in kill_sites:
            self.locks.release_lock(lock_uuid, KillCommand())

        # TODO: 7. set the current traces
        raise NotImplementedError("working on getting the latest traces for next step")


def model():

    # get our normal
    n1 = pyro.sample("n1", dist.normal, VTA(0), VTA(1))

    print("Continuing from n1 to n2")

    # our second normal
    n2 = pyro.sample("n2", dist.normal, VTA(0), VTA(1))

    # multiply our normals for whatever reason
    return n1*n2


def run_fork_model():

    # create our fork poutine
    fp = ForkPoutine(model, ForkContinueCommand)

    # run our fork trace
    fork_trace = fp()

    # get back the traces (i guess)
    return fork_trace


@multi_try(10, wait=.1)
def try_start_process(*args, **kwargs):
    p = Process(*args, **kwargs)
    p.start()
    return p


def get_batchsize_ixs(total_size, batch_size):
    return list(np.arange(0, total_size, batch_size)[1:]) + [total_size]


def main(*args, **kwargs):
    mp.set_start_method('fork')

    red = RTraces()
    rr = red.r
    print("WARNING FLUSHING ALL REDIS")
    rr.flushall()
    # rr.flushdb()

    # create num_threads == 200
    # rr.incr("num_threads", 200)
    # deeply related to process limits
    # https://apple.stackexchange.com/questions/77410/how-do-i-increase-ulimit-u-max-user-processes
    # Fixing ulimit shenangians for macbook pro:
    # this is why the processes die
    # https://blog.dekstroza.io/ulimit-shenanigans-on-osx-el-capitan/
    # no such limits on opus gpu machines :)
    call_count = 1

    # create all of our processes
    all_processes = [try_start_process(target=run_fork_model, args=[])
                     for i in range(call_count)]

    # now let's check our waiting messages. We should have one for each trace/callsite
    locks = RLock()
    traces = RTraces()
    messages = RMessages()
    clones = RPairs()
    time.sleep(.1)
    ak = locks.get_all_keys()
    ts = traces.get_all_keys()
    # locks.release_lock(ts[-1], LogPdfCommand())
    # locks.release_lock(ts[-1], CloneCommand(3))
    ck = clones.get_all_keys()
    bb()
    # locks.release_lock(ts[-1], ResampleForkContinueCommand())
    logp = loads(messages.get_value(ts[-1]))




    list(map(lambda x: x.join(), all_processes))
    print("sleep to wait for final print outs")
    time.sleep(1.5)
    print("All processes forked, caught, released and killed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
