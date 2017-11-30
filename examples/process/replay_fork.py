import torch
import signal
from os import fork, _exit, kill, getpid
# import torch.multiprocessing as mp
# from multiprocessing import Queue
from cloudpickle import loads, dumps
from pyro.util import ng_ones, ng_zeros

import pyro
import pyro.infer
import pyro.poutine as poutine
import pyro.distributions as dist
from numpy.random import seed, randint
from functools import partial
from redis import StrictRedis


def ipc_get(pid, db=0):
    r = StrictRedis(host='localhost', db=db)
    val = r.get(pid)
    r.delete(pid)
    r.connection_pool.disconnect()
    return val


def ipc_put(pid, val, db=0):
    r = StrictRedis(host='localhost', db=db)
    r.set(pid, val)
    r.connection_pool.disconnect()


def _swap_stack(frame, new_stack):
    # find this poutine's position in the stack,
    # then remove it and everything above it in the stack.
    # assert frame in pyro._PYRO_STACK, "frame must exist in the stack to be swapped!"

    # here we remove the frames above the current
    if frame in pyro._PYRO_STACK:
        loc = pyro._PYRO_STACK.index(frame)

        # we are replacing this part of the stack, so we actually inherit
        # from that object (e.g. a resume trace)
        pyro._PYRO_STACK[loc].update(new_stack[0])

        for i in range(loc+1, len(pyro._PYRO_STACK)):
            pyro._PYRO_STACK.pop()

    # then we append our new stack
    pyro._PYRO_STACK.extend(new_stack[1:])


def _merge_stack(frame, new_stack):
    assert frame in pyro._PYRO_STACK, "frame must exist in stack to start merge"

    loc = pyro._PYRO_STACK.index(frame)

    assert loc + len(new_stack) == len(pyro._PYRO_STACK), \
        "length of remaining stack must match the merge operation"

    for new_ix, frame_ix in enumerate(range(loc, len(pyro._PYRO_STACK))):
        pyro._PYRO_STACK[frame_ix].update(new_stack[new_ix])


def _fork_and_wake(msg, parent_fct, child_fct):

    def _handle_cont(*args):
        pass

    def _handle_int(*args):
        _exit(0)
        pass

    # snapshot
    print("forking at {}: {}".format(msg["name"], getpid()))

    pid = fork()
    if pid:  # parent
        return parent_fct(msg, pid)
    else:
        print("CHILD GOING TO SLEEP {}".format(getpid()))
        signal.signal(signal.SIGCONT, _handle_cont)
        signal.signal(signal.SIGINT, _handle_int)
        signal.pause()

        signal.signal(signal.SIGINT, signal.default_int_handler)

        # read from queue
        pt_ctx = loads(ipc_get(getpid()))
        # pt_ctx = loads(R().get())

        # wake up and then call the child fct with the object we received
        return child_fct(msg, getpid(), pt_ctx)


class NightmarePoutine(poutine.TracePoutine):

    def __init__(self, fn, trace=None, site=None, *args, **kwargs):
        self.is_child = False
        self.trace = trace
        self.site = site
        self.pid = getpid()
        self.is_resume = self.trace is not None
        super(NightmarePoutine, self).__init__(fn, *args, **kwargs)

    def _parent_sample(self, sample, msg, pid, *args, **kwargs):

        if "pid" in msg:
            # kill the old pid
            kill(msg["pid"], signal.SIGINT)

        msg["pid"] = pid

        # what do parents do? normal behavior
        if sample:
            return super(NightmarePoutine, self)._pyro_sample(msg)

    def set_seed(self):
        seed()
        torch.manual_seed(randint(10000000))

    # we woke up!
    def _child_sample(self, msg, pid, pt_ctx, *args, **kwargs):

        # call the parent function and don't take a sample at this site
        parent_no_sample = partial(self._parent_sample, False)

        # before we do anything, fork
        _fork_and_wake(msg, parent_no_sample, self._child_sample)

        self.is_child = True
        self.set_seed()
        _swap_stack(self, pt_ctx['stack'])

        # anything marked as done is lying to us!
        msg["done"] = False
        return super(NightmarePoutine, self)._pyro_sample(msg)

    def update(self, snapshot):
        print("UPDATE FROM SNAP {}".format(snapshot))
        # this is the pid for the master thread
        self.pid = snapshot.pid
        self.site = snapshot.site
        return super(NightmarePoutine, self).update(snapshot)

    def _pyro_sample(self, msg):
        # we have two regimes.
        # by default we fork
        parent_with_sample = partial(self._parent_sample, True)
        return _fork_and_wake(msg, parent_with_sample, self._child_sample)

    def __exit__(self, *args, **kwargs):

        print("EXITING RESUME as {} in {}".format("child" if self.is_child else 'parent', getpid()))

        # if child:
        # ran until exit
        # push current stack to queue
        if self.is_child:
            cur_stack = list(pyro._PYRO_STACK)

            super(NightmarePoutine, self).__exit__(*args, **kwargs)

            print("ABOUT TO EXIT CHILD WITH TRACE {}".format(self.trace.nodes()))  # data=True)))
            # send back the pid and the stack
            stack_obj = {
                         'stack': cur_stack,
                         'trace': self.trace,
                         'value': self.ret_value}  # self.trace.node['_RETURN']['value']}

            # send back the stack
            # self.trace.graph["queue"].put(dumps(stack_obj))
            # R().put(dumps(stack_obj))
            ipc_put(self.pid, dumps(stack_obj))

            def _handle_cont():
                pass

            print("WAKING UP THE PARENT: {}".format(self.pid))
            # wake up our parent
            signal.signal(signal.SIGCONT, _handle_cont)
            kill(self.pid, signal.SIGCONT)

            # kill child
            _exit(0)
        else:
            print("PARENT EXIT {}".format(getpid()))
            # parent and child unwind (add _RETURN statement)
            if "_RETURN" in self.trace:
                self.trace.remove_node("_RETURN")

            return super(NightmarePoutine, self).__exit__(*args, **kwargs)

    def _init_trace(self, *args, **kwargs):
        if not self.is_resume:
            print("No Trace, creating one")
            return super(NightmarePoutine, self)._init_trace(*args, **kwargs)

        # have existing trace
        print("Resume ignores trace creation")

    def _post_site_trace(self):
        post_site = self.trace.copy()
        site_loc = list(post_site.nodes()).index(self.site)

        # remove all post-site traces
        post_site.remove_nodes_from([n for n_ix, n in enumerate(post_site.nodes())
                                     if n_ix >= site_loc])

        return post_site

    def __enter__(self, *args, **kwargs):

        # when we resume and enter
        if self.is_resume:

            # make sure we install self on the stack
            super(NightmarePoutine, self).__enter__(*args, **kwargs)

            def _handle_cont(*args):
                pass

            # parent resume:
            # get the pid
            pid = self.trace.node[self.site]["pid"]

            # push stack onto queue
            # send signal to resume
            frame_loc = pyro._PYRO_STACK.index(self)

            self.trace = self._post_site_trace()
            stack_obj = {'pid': getpid(), 'stack': pyro._PYRO_STACK[frame_loc:]}

            # send our stack across the queue
            # R().put(dumps(stack_obj))
            ipc_put(pid, dumps(stack_obj))
            # self.trace = orig_trace
            # self.trace.graph["queue"].put(dumps(stack_obj))

            # on the parent, ignore the continue signal
            signal.signal(signal.SIGCONT, _handle_cont)

            # wakes up the child, which will read from the queue
            kill(pid, signal.SIGCONT)

            print("PARENT ASLEEP {} is child? {}".format(getpid(), self.is_child))
            # wait on queue for response
            signal.pause()

            print("PARENT AWAKE {} is child? {}".format(getpid(), self.is_child))

            # get the stack
            # pt_ctx = loads(R().get())
            pt_ctx = loads(ipc_get(getpid()))

            # use response to swap stack
            print("MASTER STACK MERGE")
            _merge_stack(self, pt_ctx['stack'])

        else:
            return super(NightmarePoutine, self).__enter__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.is_resume:
            print("RESUME MASTER CALL {}".format(getpid()))
            # self.is_child = False
            with self:
                return self.ret_value
            print("FINISHED RESUME {}".format(self.deadly))

        else:
            print("SNAPSHOT CALL")
            return super(NightmarePoutine, self).__call__(*args, **kwargs)


def main():
    print("model sample a")
    pyro.sample("a", dist.normal, ng_zeros(1), ng_ones(1))
    print("model sample b")
    pyro.sample("b", dist.normal, ng_zeros(1), ng_ones(1))
    print("model sample c")
    c = pyro.sample("c", dist.normal, ng_zeros(1), ng_ones(1))
    return c


def kill_trace(tr):
    for site_name, site in tr.nodes(data=True):
        if 'pid' in site:
            print("Killing {} at {}".format(site['pid'], site_name))
            try:
                kill(site['pid'], signal.SIGINT)
            except ProcessLookupError:  # noqa: F821
                pass


if __name__ == "__main__":

    # get our trace, with snapshots at each sample statement
    trace = NightmarePoutine(main).get_trace()

    # modify the trace, and replace
    print("snapshots: {}".format(trace.nodes(data='pid')))

    def pt(tr):
        return "{}".format(list(map(lambda x: (x[0], x[1].data[0] if x[1] is not None else ''),
                                    tr.nodes(data='value'))))

    # proposal_trace = poutine.trace(main).get_trace()
    # proposal_trace.node["b"]['value'] = pyro.sample("b", dist.normal, ng_zeros(1), ng_ones(1))

    # resume from site b, continuing till end
    res_pt = NightmarePoutine(main, trace, "b")  # , proposal_trace)

    # we want to resume twice from the same point
    print("Resuming twice from original trace at site b")
    # post_trace_1 = poutine.trace(res_pt).get_trace()
    print("\n\nSTART RESUME TRACE 1")
    post_trace_1 = (res_pt).get_trace()
    print("FINISHED RESUME TRACE 1\n\n")
    res_pt.site = "a"
    # post_trace_2 = (res_pt).get_trace()
    print("\n\nSTART RESUME TRACE 2")
    post_trace_2 = (res_pt).get_trace()
    print("FINISHED RESUME TRACE 2\n\n")
    res_pt.site = "c"
    post_trace_3 = (res_pt).get_trace()
    print("Expecting value b/c to be different")

    print("Original trace {}".format(pt(trace)))
    # print("Proposal trace {}".format(pt(proposal_trace)))
    print("New trace_1 {}".format(pt(post_trace_1)))
    print("New trace_2 {}".format(pt(post_trace_2)))
    print("New trace_3 {}".format(pt(post_trace_3)))

    print("Terminating trace_1 {}")
    kill_trace(post_trace_1)
    print("Terminating trace_2 {}")
    kill_trace(post_trace_2)
    print("Terminating trace_3 {}")
    kill_trace(post_trace_3)
    kill_trace(trace)
    print("FINISHED!")


