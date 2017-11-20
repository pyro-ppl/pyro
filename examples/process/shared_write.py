from pdb import set_trace as bb
import os
import multiprocessing as mp
import numpy as np
from uuid import uuid4
import functools
import redis
from multiprocessing import Process
from numpy.random import seed

# Forking multiprocessing:
# https://github.com/python/cpython/blob/3972628de3d569c88451a2a176a1c94d8822b8a6/Lib/multiprocessing/popen_fork.py


# Managers:
# https://github.com/python/cpython/blob/3972628de3d569c88451a2a176a1c94d8822b8a6/Lib/multiprocessing/context.py
def get_uuid():
    return uuid4().hex


# https://stackoverflow.com/questions/36295766/using-yield-twice-in-contextmanager
def multi_try(retries=1):
    def _wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            for _ in range(retries + 1):  # Loop retries + first attempt
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    e = exc
                    pass
                    # print("fct call failed {}".format(e.args))

            raise Exception("all fct calls failed {}".format(e.args))
        return wrapped
    return _wrapper


@multi_try(20)
def connect_add_message(msg):
    print("Init redis")
    red = R()
    uuid = get_uuid()
    print("Set redis {} - {}".format(uuid, msg))
    # red.r.set(uuid, msg)
    red.r.lpush('messages', msg)


@multi_try(20)
def fork():
    pid = os.fork()
    return pid


def access_shared_queue(i):
    # fork immediately
    pid = os.fork()
    # give each fork a new seed
    seed()

    if pid:
        # add our message to redis according to thread
        connect_add_message("hello master doofus {}".format(i))
        print("closing master {}".format(i))
        os._exit(0)

    else:
        # attempt to get a connection up to 20 times
        connect_add_message("hello child doofus {}".format(i))
        print("closing child {}".format(i))
        os._exit(0)


class R():
    def __init__(self):
        self.r = redis.StrictRedis(host='localhost', port=6379, db=0)


def main(*args, **kwargs):
    mp.set_start_method('fork')

    red = R()
    rr = red.r
    rr.flushdb()
    # create num_threads == 200
    # rr.incr("num_threads", 200)
    call_count = 1000
    # m = Manager()
    # bd = m.BoundedSemaphore(min(500, call_count))
    fork_size = 100
    start_ix = 0
    all_chunks = list(np.arange(0, call_count, fork_size)[1:]) + [call_count]

    bb()
    for end_ix in all_chunks:

        # launch all of our processes
        all_processes = [Process(target=access_shared_queue, args=[i])
                         for i in range(start_ix, end_ix)]

        list(map(lambda x: x.start(), all_processes))
        list(map(lambda x: x.join(), all_processes))

        # while rr.llen('messages') < call_count*2:
        #     pass

        start_ix = end_ix

    print("All messages inserted")
    mc = rr.llen('messages')
    am = rr.lrange('messages', 0, mc)
    print("First 10 messages {}".format(am[0:10]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
