from pyro.poutine import TracePoutine
from command_points import add_control_point, KillCommand
from ip_communication import multi_try, get_uuid
from ip_communication import RLock, RTraces, RPairs
from multiprocessing import Process
from os import _exit


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


def call_and_response(self, fork_id, call_method='get_trace', is_exit=None, *args, **kwargs):
    retry_interval = kwargs.pop('retry_interval', None)
    # hold our generic args
    obj = Object()
    # unique identify this call/response
    obj.uuid = get_uuid()
    # kill process? set to true
    obj.is_exit = is_exit
    # pass over trace call_method
    obj.call_method = call_method
    obj.args = args
    obj.kwargs = kwargs

    # send ourselves over via locking mechanism
    RLock.release_lock(fork_id, obj)

    # then wait synchronously on our response
    return RLock.add_lock_and_wait(obj.uuid, retry_interval=retry_interval)


def run_fork_process(fork_id=1, *args, **kwargs):
    kwargs["is_remote"] = False
    retry_interval = kwargs.pop("retry_interval", None)
    fp = ForkPoutine(*args, **kwargs)
    while True:
        # send through an object with commands on what to do
        fork_command = RLock.add_lock_and_wait(fork_id, retry_interval=retry_interval)

        # waiting on fork exit
        if fork_command.is_exit:
            _exit(0)

        # we have a command object for our fork, now time to execute
        ret_val = getattr(fp, fork_command.call_method)(*fork_command.args, **fork_command.kwargs)

        # then we release the object and start all over again
        RLock.release_lock(fork_command.uuid, ret_val)


# simple fork just takes a command and applies it at every site encountered
class ForkPoutine(TracePoutine):
    def __init__(self, fn, CommandCls, is_remote=False):
        # each trace is assigned a unique ID by default
        self.CommandCls = CommandCls
        self._trace_processes = []
        self.is_remote = is_remote
        self.trace_pairs = RPairs()
        self.trace_objects = RTraces()
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
            add_control_point(self.trace, site_name, self.CommandCls(), msg)

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
        # notice the force=True to add to existing obj
        self.trace.add_node(_R(), force=True, batch_log_pdf=self.trace.batch_log_pdf())

        # create a final control point that we can use to calculate other values
        add_control_point(self.trace, _R(), self.CommandCls(), msg)

        # all done
        return r_val

    def initialize(self, num_traces):
        self.fork_ids = ["fork_{}".format(i) for i in range(num_traces)]
        self._last_fork = 0

        # create a bunch of non-remote trace processes
        self._trace_processes += [try_start_process(target=run_fork_process, args=[self.fn, self.CommandCls])
                                  for i in range(num_traces)]

    def kill_all(self):

        # kill all of our forks and all of our traces
        for fid in self.fork_ids:
            call_and_response(fid, is_exit=True)

        # we need to collect all of the sites, and release the locks with kill commands
        open_sites = RTraces.get_all_keys()

        # kill all of the open sites
        for site_uuid in open_sites:
            RLock.release_lock(site_uuid, KillCommand())

        RTraces._clear_db()
        RLock._clear_db()
        RPairs._clear_db()

        # wait for all of the processes to rejoin
        list(map(lambda x: x.join(), self._trace_processes))
        self._trace_processes = []

    def continue_trace(self, trace, site_name, command_obj):
        # pick a site in a trace and operate on it
        site_uuid = RTraces.get_trace_key(trace.trace_uuid, site_name)

        # release the location and apply the command
        # TODO: we need to know what we're waitin for
        # assumed that we were waiting for a new trace to finish
        trace_count = len(RTraces.get_all_keys(match=_R("*")))
        existing_pairs = set(self.trace_pairs.get_pair_uuids(site_uuid))
        new_trace_count = trace_count
        RLock.release_lock(site_uuid, command_obj)

        # then we need to wait until we have an extra return object
        while new_trace_count < trace_count + 1:
            new_trace_count = len(RTraces.get_all_keys(match=_R("*")))

        # waited for the trace to come in, now we're ready to return the exact trace
        new_pairs = list(set(RPairs.get_pair_uuids(site_uuid)).difference(existing_pairs))

        assert len(new_pairs) == 1, "expecting one paired result per continue_trace operation"
        uuid_pair = RPairs.get_pair_from_key(new_pairs[0])

        # get the trace with the new name
        # TODO: this is all so clunky, should be a more call and response model like with forks
        return self.trace_objects.get_value(uuid_pair[1])

    def next_trace_process(self):
        fid = self.fork_ids[self._last_fork]
        self._last_fork += 1
        self._last_fork = self._last_fork % len(self.fork_ids)
        return fid

    def get_trace(self, *args, **kwargs):
        if self.is_remote:
            # execute the trace on a fork, sync wait for serialized results,
            # then return the trace object
            fid = self.next_trace_process()

            # call that specific fork with the lock and wait on the response
            return call_and_response(fid, 'get_trace', is_exit=False, *args, **kwargs)
        else:
            return super(ForkPoutine, self).get_trace(*args, **kwargs)
