import torch
import signal
from os import pipe, fork, fdopen, set_inheritable, _exit, close, kill, getpid
import torch.multiprocessing as mp
from multiprocessing import Queue
from cloudpickle import loads, dumps

import pyro
import pyro.infer
import pyro.poutine as poutine
import pyro.distributions as dist

from redis import StrictRedis

# queue = Queue()


class R():
    def __init__(self, db=0, *args, **kwargs):
        self.r = StrictRedis(host='localhost', db=db, port=6379, *args, **kwargs)

    def put(self, val):
        print("PUTTING ")
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


class SnapshotPoutine(poutine.TracePoutine):

    def __init__(self, *args, **kwargs):
        self.is_child = False
        super(SnapshotPoutine, self).__init__(*args, **kwargs)

    def _pyro_sample(self, msg):

        def _handle_cont(*args):
            pass

        def _handle_int(*args):
            _exit(0)
            pass

        # snapshot
        print("forking at {} - {}".format(msg["name"], getpid()))

        pid = fork()
        if pid:  # parent
            msg["pid"] = pid
            self.is_child = False

        else:  # child
            signal.signal(signal.SIGCONT, _handle_cont)
            signal.signal(signal.SIGINT, _handle_int)
            signal.pause()
            self.is_child = True
            # child resume:
            # wake up
            signal.signal(signal.SIGINT, signal.default_int_handler)

            # read from queue
            pt_ctx = loads(R().get())
            # pt_ctx = loads(self.trace.graph["queue"].get())

            # install stack from queue -- this includes running resume before continuing
            # set self.ppid as the pid of the calling thread
            self.ppid = pt_ctx['pid']
            _swap_stack(self, pt_ctx['stack'])

            # self.frame_replace = pt_ctx['stack'][1]
            self.trace = pt_ctx['stack'][0].trace
            # TMP HACK
            return pt_ctx['stack'][0]._pyro_sample(msg)
            # if pt_ctx['stack'][0].site == msg["name"]:
            #     # return self._pyro_sample(msg)
            # else:
            #     return msg["value"]
            # return msg["value"]

        # return self._pyro_sample(msg)
        # print(msg)
        return super(SnapshotPoutine, self)._pyro_sample(msg)

    def __exit__(self, *args, **kwargs):

        print("SNAP EXIT {}".format(self.is_child))
        cur_stack = list(pyro._PYRO_STACK)

        # parent and child unwind (add _RETURN statement)
        # super(SnapshotPoutine, self).__exit__(*args, **kwargs)

        # if child:
        # ran until exit
        # push current stack to queue
        if self.is_child:
            # pyro._PYRO_STACK.pop(0)

            # parent and child unwind (add _RETURN statement)
            # self.frame_replace.__exit__(*args, **kwargs)
            super(SnapshotPoutine, self).__exit__(*args, **kwargs)
            # self.frame_replace.__exit__(*args, **kwargs)

            print("ABOUT TO EXIT CHILD WITH TRACE {}".format(self.trace.nodes()))
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

            # wake up our parent
            signal.signal(signal.SIGCONT, _handle_cont)
            kill(self.ppid, signal.SIGCONT)

            # kill child
            _exit(0)
        else:
            # parent and child unwind (add _RETURN statement)
            super(SnapshotPoutine, self).__exit__(*args, **kwargs)


    def _init_trace(self, *args, **kwargs):
        print(" Snap Init trace")
        r_val = super(SnapshotPoutine, self)._init_trace(*args, **kwargs)
        # self.trace.graph["queue"] = Queue()
        return r_val


class ResumePoutine(SnapshotPoutine):

    def __init__(self, fn, trace, site):
        self.site = site
        self.trace = trace

        self.pid = getpid()
        super(ResumePoutine, self).__init__(fn)

    def _init_trace(self, *args, **kwargs):
        print(" Resume Init trace, remove to be sampled")

    def _post_site_trace(self):
        post_site = self.trace.copy()
        site_loc = list(post_site.nodes()).index(self.site)
        # remove all post-site traces
        # post_site.remove_node("_RETURN")
        post_site.remove_nodes_from([n for n_ix, n in enumerate(post_site.nodes())
                                     if n_ix >= site_loc])

        return post_site

    def __enter__(self, *args, **kwargs):
        # make sure we install self on the stack
        super(ResumePoutine, self).__enter__(*args, **kwargs)

        def _handle_cont(*args):
            pass

        # parent resume:
        # get the pid
        pid = self.trace.node[self.site]["pid"]

        # push stack onto queue
        # send signal to resume
        frame_loc = pyro._PYRO_STACK.index(self)

        orig_trace = self.trace
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

        # wait on queue for response
        signal.pause()

        # get the stack
        pt_ctx = loads(R().get())
        # pt_ctx = loads(self.trace.graph["queue"].get())

        # use response to swap stack
        _swap_stack(self, pt_ctx['stack'])

        # having replaced ourselves on the stack, we fetch the relevent return
        self.ret_value = pt_ctx['value']
        print(pt_ctx["value"])
        #
        self.trace = pt_ctx['trace']
        print("Local stack: {} \n\n incoming: {}".format(pyro._PYRO_STACK,
                                                         pt_ctx['stack']))
        print("Trace nodes: {}".format(self.trace.nodes()))

    def __call__(self, *args, **kwargs):
        print("RESUME CALL")
        # self.is_child = False
        with self:
            return self.ret_value

    def __exit__(self, *args, **kwargs):
        if "_RETURN" in self.trace:
            self.trace.remove_node("_RETURN")
        super(ResumePoutine, self).__exit__(*args, **kwargs)


def main():
    from pyro.util import ng_ones, ng_zeros
    print("model sample a")
    pyro.sample("a", dist.normal, ng_zeros(1), ng_ones(1))
    print("model sample b")
    pyro.sample("b", dist.normal, ng_zeros(1), ng_ones(1))
    print("model sample c")
    pyro.sample("c", dist.normal, ng_zeros(1), ng_ones(1))


if __name__ == "__main__":

    # get our trace, with snapshots at each sample statement
    trace = SnapshotPoutine(main).get_trace()

    print(trace.graph)

    # resume from site b, continuing till end
    res_pt = ResumePoutine(main, trace, "b")

    # we want to resume twice from the same point
    print("Resuming twice from original trace at site b")
    # post_trace_1 = poutine.trace(res_pt).get_trace()
    post_trace_1 = res_pt.get_trace()
    print("FINISHED RESUME TRACE 1")
    # res_pt.site = "c"
    # post_trace_2 = poutine.trace(res_pt).get_trace()
    post_trace_2 = res_pt.get_trace()
    print("FINISHED RESUME TRACE 2")

    print("Expecting value b/c to be different")
    print("Original trace {}".format(trace.nodes(data='pid')))
    print("New trace_1 {}".format(post_trace_1.nodes(data='pid')))
    print("New trace_2 {}".format(post_trace_2.nodes(data='pid')))



