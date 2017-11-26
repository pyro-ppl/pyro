
from pickle import dumps
from ip_communication import RLock, RTraces, RMessages, RPairs, get_uuid
from os import fork, _exit
from numpy.random import seed, randint
from torch import manual_seed


def add_control_point(trace, site_name, ctrl_behavior, *args, **kwargs):
    return ctrl_behavior.execute_control_site(trace, site_name, *args, **kwargs)


class ControlCommand():
    def __init__(self):
        pass

    def wait_on_lock(self, trace_uuid, site_name, *args, **kwargs):
        lock_uuid = RTraces.get_trace_key(trace_uuid, site_name)
        locker = RLock()
        print("Waiting on lock: {}".format(lock_uuid))
        wake_command = locker.add_lock_and_wait(lock_uuid)
        print("Released from lock: {}".format(lock_uuid))
        locker.kill_connection()
        return wake_command

    def execute_control_site(self, trace, site_name, *args, **kwargs):
        raise NotImplementedError("Abstract class")


class ForkContinueCommand(ControlCommand):

    # don't actually serialize everything and send back
    def serialize_trace(self, trace, site_name):

        # clone and continue says parent continue, child gets frozen
        trace_str = dumps(trace)

        # set the trace object in our shared redis db, then kill conn
        RTraces().set_trace(trace.trace_uuid, site_name, trace_str).kill_connection()

    # create a branch, freeze it, then continue
    def execute_control_site(self, trace, site_name, *args, **kwargs):
        print("Fork/Continue at site: {}, {}, {}".format(trace.trace_uuid, site_name, trace))

        # serialize the trace
        self.serialize_trace(trace, site_name)

        # try fork and proceed
        pid = fork()

        # continue
        if pid:
            # we're the parent
            return self.parent_fct(trace, site_name, *args, **kwargs)
        else:
            # we're the child
            return self.child_fct(trace, site_name, *args, **kwargs)

    def child_fct(self, trace, site_name, *args, **kwargs):

        # we'll wait on this, get back how to continue
        wake_command = self.wait_on_lock(trace.trace_uuid, site_name)

        # when we wake up, we'll read a new control object, then
        # execute accordingly
        assert issubclass(type(wake_command), ControlCommand), "Lock must be behavior for lock release"
        return wake_command.execute_control_site(trace, site_name, *args, **kwargs)

    def parent_fct(self, trace, site_name, *args, **kwargs):
        # parent site simply continues
        print("trace parent {} - {}".format(trace.trace_uuid, site_name))
        pass


class CloneCommand(ForkContinueCommand):
    def __init__(self, clone_count=1):
        self.clone_count = clone_count
        super(CloneCommand, self).__init__()

    # don't do anything here
    def serialize_trace(self, *args, **kwargs):
        print("")
        pass

    # when the execute command gets called, the child will branch
    # then wait at the old address
    # then the parent will fork a bunch more and store the children with a new address
    def parent_fct(self, trace, site_name, *args, **kwargs):

        # parent site will control the forking
        for i in range(self.clone_count):

            # then set our pairing
            child_trace_uuid = get_uuid()
            pair_key = RPairs.get_pair_name(trace.trace_uuid, child_trace_uuid, site_name)

            # try fork and proceed
            pid = fork()

            # if we're the
            if pid == 0:

                # from here on out, the trace has a new ID
                trace.trace_uuid = child_trace_uuid

                # add this pair to redis
                RPairs().add_pair_uuids(pair_key).kill_connection()

                # store the cloned trace
                ForkContinueCommand.serialize_trace(self,
                                                    trace,
                                                    site_name)

                # we're the child, store with new child uuid
                # must RETURN here because otherwise we'd fall through to exit command
                return self.child_fct(trace, site_name, *args, **kwargs)

        # kill the parent, we've done our job
        _exit(0)


class ResampleCloneContinueCommand(CloneCommand):
    def __init__(self, seed=None, *args, **kwargs):
        self.seed = seed
        super(ResampleCloneContinueCommand, self).__init__(*args, **kwargs)

    def child_fct(self, *args, **kwargs):
        # normally we would wait, instead we simple run a resampleforkcontinue
        # we already handle the new id, don't create uuid_on_sample
        return ResampleForkContinueCommand(self.seed, preserve_parent=False, uuid_on_sample=False)\
                .execute_control_site(*args, **kwargs)


class ApplyFunctionForkContinueCommand(ForkContinueCommand):
    def __init__(self, preserve_parent=False, uuid_on_sample=True):
        self.uuid_on_sample = uuid_on_sample
        self.preserve_parent = preserve_parent
        super(ApplyFunctionForkContinueCommand, self).__init__()

    def apply_function(self, *args, **kwargs):
        pass

    # resample then lock on site again
    def execute_control_site(self, trace, site_name, *args, **kwargs):

        # we want to preserve the parent command point
        if self.preserve_parent:
            # first we fork, then we get to continue as parent
            super(ApplyFunctionForkContinueCommand, self).execute_control_site(trace,
                                                                               site_name,
                                                                               *args, **kwargs)
        # wake up and do something
        self.apply_function(trace, site_name, *args, **kwargs)

        # we're part of a new object now
        if self.uuid_on_sample:

            # create a new sample to continue onwards
            child_trace_uuid = get_uuid()
            pair_key = RPairs.get_pair_name(trace.trace_uuid, child_trace_uuid, site_name)

            # from here on out, the trace has a new ID
            trace.trace_uuid = child_trace_uuid

            # add this pair to redis
            RPairs().add_pair_uuids(pair_key).kill_connection()

        return super(ApplyFunctionForkContinueCommand, self).execute_control_site(trace,
                                                                                  site_name,
                                                                                  *args, **kwargs)


# Function takes a sample_nudge at construction, and simply applies the nudge to the existing value
# lots of assumptions made, and no assertions or checks
class NudgeForkContinueCommand(ApplyFunctionForkContinueCommand):
    def __init__(self, sample_nudge, *args, **kwargs):
        self.sample_nudge = sample_nudge
        super(NudgeForkContinueCommand, self).__init__(*args, **kwargs)

    def apply_function(self, trace, site_name, msg):
        # wake up and add our nudge factor to the original value
        trace.add_node(msg["name"], force=True, value=msg["value"] + self.sample_nudge)


# Take a given object and resample and continue\
# this is nice because we can fork and resample
class ResampleForkContinueCommand(ApplyFunctionForkContinueCommand):

    def __init__(self, seed=None, *args, **kwargs):
        self.seed = seed
        super(ResampleForkContinueCommand, self).__init__()

    def apply_function(self, trace, site_name, msg):
        seed(self.seed)
        # set our seed manually, or use numpy random setup to seed
        manual_seed(self.seed if self.seed is not None else randint(0, 10000000))

        print("Conducting: {}".format(msg))
        # resample this site inside of pyro stack
        assert msg["type"] == "sample", "Cannot execute a resample at a non-sample node"
        print("Resampling: {}".format(site_name))
        # resample from the poutine we had going
        trace._pyro_sample(msg)


# wake up, calculate the log pdf of the trace, store it, then go back to sleep
class LogPdfCommand(ControlCommand):

    # resample then lock on site again
    def execute_control_site(self, trace, site_name, *args, **kwargs):

        log_uuid = RTraces.get_trace_key(trace.trace_uuid, site_name)

        # use the trace to calculate log_pdf
        trace_pdf = trace.batch_log_pdf()

        # do something else with trace?

        # set our message, then wait on any responses
        RMessages().set_msg(log_uuid, dumps({'log_pdf': trace_pdf})).kill_connection()

        # we'll wait on this, get back how to continue
        wake_command = self.wait_on_lock(trace.trace_uuid, site_name)

        # when we wake up, we'll read a new control object, then
        # execute accordingly
        assert issubclass(type(wake_command), ControlCommand), "Lock must be behavior for lock release"
        return wake_command.execute_control_site(trace, site_name, *args, **kwargs)


# wake up, calculate the log pdf of the trace, store it, then go back to sleep
class KillCommand(ControlCommand):

    # resample then lock on site again
    def execute_control_site(self, *args, **kwargs):
        # simple as clearing and exiting
        # TODO: remove self from redis?
        _exit(0)
