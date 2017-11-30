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

# queue = Queue()


class R():
    def __init__(self, db=0, *args, **kwargs):
        self.r = StrictRedis(host='localhost', db=db, port=6379, *args, **kwargs)

    def put(self, val):
        self.r.set("key", val)

    def get(self):
        return self.r.get("key")


def _swap_stack(frame, new_stack):
    # find this poutine's position in the stack,
    # then remove it and everything above it in the stack.
    # assert frame in pyro._PYRO_STACK, "frame must exist in the stack to be swapped!"

    # here we remove the frames above the current
    if frame in pyro._PYRO_STACK:
        loc = pyro._PYRO_STACK.index(frame)
        for i in range(loc, len(pyro._PYRO_STACK)):
            pyro._PYRO_STACK.pop()

    # then we append our new stack
    pyro._PYRO_STACK.extend(new_stack)


def _merge_stack(frame, new_stack):
    assert frame in pyro._PYRO_STACK, "frame must exist in stack to start merge"

    loc = pyro._PYRO_STACK.index(frame)
    assert loc + len(new_stack) == len(pyro._PYRO_STACK), "length of remaining stack must match the merge operation"

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
        pt_ctx = loads(R().get())

        # wake up and then call the child fct with the object we received
        return child_fct(msg, getpid(), pt_ctx)


class NightmarePoutine(poutine.TracePoutine):

    def __init__(self, fn, trace=None, site=None, *args, **kwargs):
        self.is_child = False
        self.is_parent_child = False
        self.trace = trace
        self.site = site
        self.pid = getpid()
        self.is_resume = self.trace is not None
        super(NightmarePoutine, self).__init__(fn, *args, **kwargs)

    def _parent_sample(self, msg, pid, *args, **kwargs):
        msg["pid"] = pid
        self.is_child = False
        # what do parents do? normal behavior
        return super(NightmarePoutine, self)._pyro_sample(msg)

    # we woke up!
    def _child_sample(self, msg, pid, pt_ctx, *args, **kwargs):
        def p(msg, pid, *args, **kwargs):
            msg["pid"] = pid
            pass

        # before we do anything, fork
        print("CHILD AWAKE {} : {}".format(getpid(), msg["name"]))
        _fork_and_wake(msg, p, self._child_sample)

        print("PARENT-CHILD AWAKE {} : {}".format(getpid(), msg["name"]))
        self.is_parent_child = True
        self.is_child = False
        seed()
        torch.manual_seed(randint(10000000))
        # child resume:
        # wake up
        self.ppid = pt_ctx['pid']
        self.replace_frame = pt_ctx['stack'][0]

        # don't swap the stack send in
        self.trace = pt_ctx['stack'][0].trace
        # self.is_resume = pt_ctx['stack'][0].is_resume
        # self.is_child = pt_ctx['stack'][0].is_child
        print("Stack {}".format([self] + pt_ctx['stack'][1:]))
        # _swap_stack(self, pt_ctx['stack'])
        _swap_stack(self, [self] + pt_ctx['stack'][1:])

        print("FRAME COMPARE - REPL_FRAME")
        print(self.replace_frame)
        print("FRAME COMPARE - SELF")
        print(self)

        return super(NightmarePoutine, self)._pyro_sample(msg)

        # self.is_child = False
        # return self._pyro_sample(msg)
        # def pkill(msg, pid, *args, **kwargs):
        #     msg["pid"] = pid
        #     self.is_child = False
        #     self.is_resume = True
        #     self.deadly = True
        #     return super(NightmarePoutine, self)._pyro_sample(msg)

        # return self.replace_frame._pyro_sample(msg)

        # def p(msg, pid, *args):
        #     msg["pid"] = pid
        #     self.is_parent_child = True
        #     print("To resume: {}".format(self.ppid))
        #     # what do parents do? normal behavior
        #     return super(NightmarePoutine, self)._pyro_sample(msg)

        # def c(msg, pid, *args, **kwargs):
        #     print("C LOOP {}:{}".format(msg["name"], pid))
        #     return self._child_sample(msg, pid, *args, **kwargs)

        # return _fork_and_wake(msg, p, c)

        #     # self.is_child = False
        #     # self.is_resume = True
        # # return _fork_and_wake(msg, self._parent_sample, self._child_sample)
        # return _fork_and_wake(msg, p, c)

        # what do parents do? normal behavior
        # return self.replace_frame._pyro_sample(msg)

    def __str__(self):
        return "CHILD {} - RESUME {} - TRACE {}".format(self.is_child, self.is_resume, self.trace.nodes())

    def _pyro_sample(self, msg):
        # we have two regimes.
        # by default we fork
        return _fork_and_wake(msg, self._parent_sample, self._child_sample)

    # def get_trace(self, *args, **kwargs):
    #     self.deadly = False
    #     r_val = super(NightmarePoutine, self).get_trace(*args, **kwargs)
    #     print("GET TRACE RETURN. ARE WE DEADLY? {}".format(self.deadly))
    #     if self.deadly:
    #         _exit(0)
    #     return r_val

    def __exit__(self, *args, **kwargs):

        print("EXITING RESUME as {} in {}".format("child" if self.is_child else 'parent', getpid()))

        # if child:
        # ran until exit
        # push current stack to queue
        if self.is_child or self.is_parent_child:
            cur_stack = list(pyro._PYRO_STACK)
            super(NightmarePoutine, self).__exit__(*args, **kwargs)

            # self.replace_frame.ret_value = self.ret_value
            # self.replace_frame.__exit__(*args, **kwargs)

            # self.frame_replace.__exit__(*args, **kwargs)

            print("ABOUT TO EXIT CHILD WITH TRACE {}".format(self.trace.nodes()))
            print("ALT TRACE {}".format(self.trace.nodes()))
            print("rv TRACE {}".format(self.ret_value))
            # send back the pid and the stack
            stack_obj = {
                         'stack': cur_stack,
                         'trace': self.trace,
                         'value': self.ret_value}  # self.trace.node['_RETURN']['value']}

            # send back the stack
            # self.trace.graph["queue"].put(dumps(stack_obj))
            R().put(dumps(stack_obj))

            def _handle_cont():
                pass

            print("WAKING UP THE PARENT: {}".format(self.ppid))
            # wake up our parent
            signal.signal(signal.SIGCONT, _handle_cont)
            kill(self.ppid, signal.SIGCONT)

            # kill child
            _exit(0)
        else:
            print("PARENT EXIT {}".format(getpid()))
            # parent and child unwind (add _RETURN statement)
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
        # post_site.remove_node("_RETURN")
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
            R().put(dumps(stack_obj))
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
            pt_ctx = loads(R().get())
            # pt_ctx = loads(self.trace.graph["queue"].get())
            # print(pt_ctx)
            # print(pt_ctx['stack'])
            # use response to swap stack
            # _merge_stack(self, pt_ctx['stack'])

            self.trace = pt_ctx['stack'][0].trace
            self.ret_value = pt_ctx['stack'][0].ret_value
            self.trace.remove_node('_RETURN')
            _swap_stack(self, [self] + pt_ctx['stack'][1:])

            # having replaced ourselves on the stack, we fetch the relevent return
            # self.ret_value = pt_ctx['value']
            # print(pt_ctx["value"])
            # #
            # self.trace = pt_ctx['trace']
            print("Local stack: {} \n\n incoming: {}".format(pyro._PYRO_STACK,
                                                             pt_ctx['stack']))
            print("Trace nodes: {}".format(self.trace.nodes()))
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


if __name__ == "__main__":

    # get our trace, with snapshots at each sample statement
    trace = NightmarePoutine(main).get_trace()

    # modify the trace, and replace
    trace.node["b"]['value'] = pyro.sample("b", dist.normal, ng_zeros(1), ng_ones(1))
    print("snapshots: {}".format(trace.nodes(data='pid')))

    # resume from site b, continuing till end
    res_pt = NightmarePoutine(main, trace, "b")

    # we want to resume twice from the same point
    print("Resuming twice from original trace at site b")
    # post_trace_1 = poutine.trace(res_pt).get_trace()
    post_trace_1 = res_pt.get_trace()
    print("FINISHED RESUME TRACE 1")
    # res_pt.site = "c"
    # post_trace_2 = poutine.trace(res_pt).get_trace()
    post_trace_2 = res_pt.get_trace()
    print("FINISHED RESUME TRACE 2")
    res_pt.site = "c"
    post_trace_3 = res_pt.get_trace()
    print("Expecting value b/c to be different")

    def pt(tr):
        return "{}".format(list(map(lambda x: (x[0], x[1].data[0] if x[1] is not None else ''),
                                    tr.nodes(data='value'))))

    print("Original trace {}".format(pt(trace)))
    print("New trace_1 {}".format(pt(post_trace_1)))
    print("New trace_2 {}".format(pt(post_trace_2)))
    print("New trace_3 {}".format(pt(post_trace_3)))



