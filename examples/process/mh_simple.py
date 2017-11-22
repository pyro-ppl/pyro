from pdb import set_trace as bb
import os
import multiprocessing as mp
import numpy as np
import functools
import redis
from multiprocessing import Process
from numpy.random import seed
import time
import torch
import pyro
from pyro.poutine import TracePoutine, Poutine
import pyro.distributions as dist
from pickle import loads
from torch import Tensor as T
from torch.autograd import Variable as V
from command_points import add_control_point, ForkContinueCommand, LogPdfCommand
from command_points import ResampleForkContinueCommand
from ip_communication import get_uuid, multi_try, RTraces, RLock, RMessages


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

        # create a final control point that we can use to calculate other values
        add_control_point(self.trace_uuid, "_RETURN", self, self.CommandCls(), msg)

        # all done
        return r_val


def model():

    # get our normal
    n1 = pyro.sample("n1", dist.normal, VTA(0), VTA(1))

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
    time.sleep(.1)
    ak = locks.get_all_keys()
    ts = traces.get_all_keys()
    # locks.release_lock(ts[-1], LogPdfCommand())
    locks.release_lock(ts[-1], ResampleForkContinueCommand())
    bb()
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
