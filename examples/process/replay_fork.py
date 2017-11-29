import torch
import signal
import os
import torch.multiprocessing as mp
import cloudpickle as pickle

import pyro
import pyro.infer
import pyro.poutine as poutine
import pyro.distributions as dist


class SnapshotPoutine(poutine.TracePoutine):

    def _pyro_sample(self, msg):

        def _handle_cont(*args):
            pass

        def _handle_int(*args):
            os._exit(0)
            pass

        # snapshot
        read_fd, write_fd = os.pipe()
        os.set_inheritable(read_fd, True)
        pid = os.fork()
        if pid:  # parent
            msg["pid"] = pid
            msg["pipe"] = write_fd
        else:  # child
            signal.signal(signal.SIGCONT, _handle_cont)
            signal.signal(signal.SIGINT, _handle_int)
            signal.pause()
            signal.signal(signal.SIGINT, signal.default_int_handler)

        return super(SnapshotPoutine, self)._pyro_sample(msg)

    def _receive_stack(self):
        pass

    def _swap_stack(self):
        pass


class ResumePoutine(poutine.Poutine):

    def __init__(self, fn, trace, site):
        self.site = site
        self.trace = trace
        super(ResumePoutine, self).__init__(fn)

    def __call__(self, *args, **kwargs):

        pid = self.trace[self.site]["pid"]
        pipe = self.trace[self.site]["pipe"]

        self._swap_stack(pid, pipe)

        def _handle_child(*args):
            pass

        def _handle_cont(*args):
            pass

        signal.signal(signal.SIGCHLD, _handle_child)
        signal.signal(signal.SIGCONT, _handle_cont)
        os.kill(pid, signal.SIGCONT)
        signal.pause()

    def _send_stack(self, pid, pipe):
        stack_string = pickle.dumps(pyro._PYRO_STACK)
        pipe.send(stack_string)

    def _receive_stack(self, pipe):
        return pickle.loads(pipe.receive())

    def _swap_stack(self, new_stack):
        pass
