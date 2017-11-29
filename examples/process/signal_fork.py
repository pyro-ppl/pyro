# we use signal to execute forks, not the original redis
from multiprocessing import Queue, Process
from uuid import uuid4
from os import fork, _exit, kill, waitpid, getpid, pipe
from os import close, fdopen
import signal
from numpy.random import choice
from collections import OrderedDict
from pickle import dumps, loads

queue = Queue()


def get_uuid():
    return uuid4().hex


# Inspired by: https://github.com/TomOnTime/timetravelpdb/blob/master/timetravelpdb.py#L163
def signal_fork():

    def _handle_cont(self, foo):
        print('CONTINUE CHILD {}'.format(getpid()))
        pass

    def _handle_int(self, foo):
        print('INT')
        _exit(0)
        pass

    r, w = pipe()
    print("SIGNAL FORKING {}".format(getpid()))
    pid = fork()
    if pid:
        print("PARENT EXECUTION {}".format(pid))
        close(r)
        # basically just store our pid
        return (pid, w)
    else:
        close(w)
        # child? split from parent
        print("CHILD SLEEPING {}".format(getpid()))
        signal.signal(signal.SIGCONT, _handle_cont)
        signal.signal(signal.SIGINT, _handle_int)
        signal.pause()

        # read some context
        r = fdopen(r)

        ctx = r.read()
        print("CHILD CONTEXT {}".format(ctx))
        r.close()

        print("CHILD WAKING {}".format(getpid()))
        signal.signal(signal.SIGINT, signal.default_int_handler)
        return signal_fork()


def main():
    trace = OrderedDict()
    pid_pipes = {}

    print("Forking")
    for ix in range(5):
        pid, pipe_id = signal_fork()
        trace["loc_{}".format(ix)] = {'pid': pid, 'pipe': pipe_id, 'uuid': get_uuid()}
        pid_pipes[pid] = pipe_id

    # store the trace of the parent
    print("Trace finished, pushing to queue")

    # def _continue(self, foo):
    #     print("UP DONE ; CONTINUE")

    # signal.signal(signal.SIGCONT, _continue)
    # kill(getppid(), signal.SIGCONT)
    write_pipe = pid_pipes[getpid()]
    w = fdopen(write_pipe, 'w')
    w.write(dumps(trace))
    w.close()
    # queue.put(trace)


if __name__ == "__main__":

    p = Process(target=main)
    p.start()

    # we wait on the first trace
    t1 = queue.get()


    print("INITIAL TRACE {}".format(t1))

    for n in range(3):
        # select a random location
        r_site = choice(list(t1.keys()))

        # get the pid to resum
        print("RESUME FROM {}".format(r_site))
        pid = t1[r_site]['pid']

        print("RESUMING PROCESS {}: {}".format(n, pid))
        # send signal to resume
        kill(pid, signal.SIGCONT)

        print("CHILD LAUNCH, WAIT ON QUEUE")
        # wait on the queue to deliver us a new return trace
        t1 = queue.get()

        print("UPDATED TRACE {}".format(t1))

    # let's kill everything!
    p.join()
    print("All done!")
