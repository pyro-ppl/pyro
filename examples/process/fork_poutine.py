from pdb import set_trace as bb
from pyro.poutine import TracePoutine
from pyro.util import get_uuid
from command_points import add_control_point, KillCommand
from ip_communication import multi_try
from ip_communication import RLock, RTraces, RPairs
from multiprocessing import Process
from os import _exit
from pickle import loads


def _R(val=''):
    return "{}_RETURN".format(val)


@multi_try(10, wait=.1)
def try_start_process(*args, **kwargs):
    p = Process(*args, **kwargs)
    p.start()
    return p


class Object(object):
    def __init__(self):
        pass

# refactors:
# consult eli on trace and replay
# futures interface
# renaming/killing
# get rid of some of the cruft
# add non-fork MH

# naming:
# poutine.resume?

# testing:
# logs in redis for each fork created: sequentially with pids - grab log and test for invariants
# ^ process tracing will be useful
# speed/performance
# neeraj benchmark profiles

# fork strat:
# MH base example in trace/replay
# fork logic + MH separate


def call_and_response(locks, fork_id, call_method='get_trace', *args, **kwargs):
    retry_interval = kwargs.pop('retry_interval', .1)
    # hold our generic args
    obj = Object()
    # unique identify this call/response
    obj.uuid = get_uuid()
    # kill process? set to true
    obj.is_exit = kwargs.pop("is_exit", False)
    # pass over trace call_method
    obj.call_method = call_method
    obj.args = args
    obj.kwargs = kwargs

    # send ourselves over via locking mechanism
    locks.release_lock(fork_id, obj)

    # then wait synchronously on our response
    return locks.add_lock_and_wait(obj.uuid, retry_interval=retry_interval)


def run_fork_process(fork_id, *args, **kwargs):
    kwargs["is_remote"] = False
    retry_interval = kwargs.pop("retry_interval", .1)

    fp = ForkPoutine(*args, **kwargs)
    fp.locks.release_lock(fork_id + "_init", {})

    # send through an object with commands on what to do
    print("Forking waiting on fid: {}".format(fork_id))
    fork_command = RLock().add_lock_and_wait(fork_id, retry_interval=retry_interval)

    # waiting on fork exit
    if fork_command.is_exit:
        print("killing fork: {}".format(fork_id))
        _exit(0)

    print("Thread executing {}:{}".format(fork_id, fork_command.call_method))
    # we have a command object for our fork, now time to execute
    ret_val = getattr(fp, fork_command.call_method)(*fork_command.args, **fork_command.kwargs)
    # print("FINISHED executing {}:{} - {}".format(fork_id, fp.trace.trace_uuid, fork_command.call_method))

    # then we release the object and start all over again
    RLock().release_lock(fork_command.uuid, ret_val).kill_connection()

    _exit(0)

    # rr = RLock()
    # if rr.r.exists(fork_command.uuid):  # llen(fork_command.uuid) > 0:
    #     print("dup post {} rv, exiting".format(fork_command.call_method))
    #     rr.kill_connection()
    #     _exit(0)

    # # then we release the object and start all over again
    # rr.release_lock(fork_command.uuid, ret_val).kill_connection()


# simple fork just takes a command and applies it at every site encountered
class ForkPoutine(TracePoutine):
    def __init__(self, fn, CommandCls, is_remote=True):
        # each trace is assigned a unique ID by default
        self.CommandCls = CommandCls
        self._trace_processes = []
        self.is_remote = is_remote
        self.trace_pairs = RPairs()
        self.trace_objects = RTraces()
        self.locks = RLock()
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
            add_control_point(self.trace, site_name, self.CommandCls(), self.trace.node[site_name])

            # get the value of the node, whatever it's set to now
            return self.trace.node[site_name]['value']

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
        # notice the force=True to add to existing obj
        self.trace.add_node(_R(), force=True, type="return", batch_log_pdf=self.trace.batch_log_pdf())

        # create a final control point that we can use to calculate other values
        add_control_point(self.trace, _R(), self.CommandCls(), msg)

        # all done
        return r_val

    def initialize(self, num_traces):
        self.fork_ids = ["fork_{}".format(i) for i in range(num_traces)]
        self._last_fork = 0

        # create a bunch of non-remote trace processes
        self._trace_processes += [try_start_process(target=run_fork_process, args=[fid, self.fn, self.CommandCls])
                                  for fid in self.fork_ids]

        for fid in self.fork_ids:
            # wait for each of the forks to initialize
            self.locks.add_lock_and_wait(fid + "_init")
        print("ALL FORKS INIT AND RUNNING")
        return self

    def kill_trace(self, trace, post_site):
        # issue kill command to all objects after the post_site
        post_seen = False
        kill_sites = []
        for nid, n in trace.nodes(data=True):
            if nid == post_site:
                post_seen = True

            if post_seen:
                kill_sites.append(RTraces.get_trace_key(trace.trace_uuid, nid))

        # print("Killing traces: {}".format(kill_sites))

        # kill all of the open sites
        for site_uuid in kill_sites:
            self.locks.release_lock(site_uuid, KillCommand())

    def kill_all(self):

        # kill all of our forks and all of our traces
        for fid in self.fork_ids:
            call_and_response(self.locks, fid, is_exit=True)

        # we need to collect all of the sites, and release the locks with kill commands
        open_sites = self.trace_objects.get_all_keys()

        # kill all of the open sites
        for site_uuid in open_sites:
            self.locks.release_lock(site_uuid, KillCommand())

        self.trace_pairs._clear_db()
        self.locks._clear_db()
        self.trace_objects._clear_db()

        # wait for all of the processes to rejoin
        list(map(lambda x: x.join(), self._trace_processes))
        self._trace_processes = []

        return self

    def find_relevant_lock(self, trace_uuid, site_name):
        site_uuid = RTraces.get_trace_key(trace_uuid, site_name)
        if not self.trace_objects.r.exists(site_uuid):
            inv_match = RPairs.get_pair_from_key(self.trace_pairs.get_inv_pair_uuids(trace_uuid)[0])
            return self.find_relevant_lock(inv_match[0], site_name)
        else:
            return trace_uuid, site_name

    def continue_trace(self, trace, site_name, command_obj):

        # get the correct location to release and watch
        rel_trace_uuid, rel_site_name = self.find_relevant_lock(trace.trace_uuid, site_name)
        site_uuid = RTraces.get_trace_key(rel_trace_uuid, rel_site_name)
        # print("\tCONTINUING TRACE: {}".format(site_uuid))

        # release the location and apply the command
        # TODO: we need to know what we're waitin for
        # assumed that we were waiting for a new trace to finish
        trace_count = len(self.trace_objects.get_all_keys(match=_R("*")))
        existing_pairs = set(self.trace_pairs.get_pair_uuids(rel_trace_uuid))
        new_trace_count = trace_count

        self.locks.release_lock(site_uuid, command_obj)

        # then we need to wait until we have an extra return object
        # print("\tWAITING ON TRACE: {}".format(site_uuid))

        while new_trace_count < trace_count + 1:
            new_trace_count = len(self.trace_objects.get_all_keys(match=_R("*")))

        # print("\tGETTING TRACE RESULTS: {}".format(site_uuid))

        # waited for the trace to come in, now we're ready to return the exact trace
        new_pairs = list(set(self.trace_pairs.get_pair_uuids(rel_trace_uuid)).difference(existing_pairs))

        assert len(new_pairs) == 1, "expecting one paired result per continue_trace operation"
        uuid_pair = RPairs.get_pair_from_key(new_pairs[0])
        ret_trace_site = RTraces.get_trace_key(uuid_pair[1], _R())
        # get the trace with the new name
        # TODO: this is all so clunky, should be a more call and response model like with forks
        return loads(self.trace_objects.get_value(ret_trace_site))

    def next_trace_process(self):
        fid = self.fork_ids[self._last_fork]
        self._last_fork += 1
        self._last_fork = self._last_fork % len(self.fork_ids)
        return fid

    def get_trace(self, *args, **kwargs):
        print("Summoning trace remotely: {}".format(self.is_remote))
        if self.is_remote:
            # execute the trace on a fork, sync wait for serialized results,
            # then return the trace object
            fid = self.next_trace_process()
            # print("Executing remote fork: {}".format(fid))
            # call that specific fork with the lock and wait on the response
            trace_resp = call_and_response(self.locks, fid, 'get_trace', *args, **kwargs)
            print("\tSYNC FINISHED GET TRACE")
            return trace_resp
        else:
            tt = super(ForkPoutine, self).get_trace(*args, **kwargs)
            # print("cur tid: {}, self : {}".format(tt.trace_uuid, self.trace.trace_uuid))
            return tt
