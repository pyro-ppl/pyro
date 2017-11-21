from pdb import set_trace as bb
import os
import multiprocessing as mp
import numpy as np
from uuid import uuid4
import functools
import redis
from multiprocessing import Process
from numpy.random import seed
import time

# Forking multiprocessing:
# https://github.com/python/cpython/blob/3972628de3d569c88451a2a176a1c94d8822b8a6/Lib/multiprocessing/popen_fork.py


# Managers:
# https://github.com/python/cpython/blob/3972628de3d569c88451a2a176a1c94d8822b8a6/Lib/multiprocessing/context.py
def get_uuid():
    return uuid4().hex


# https://stackoverflow.com/questions/36295766/using-yield-twice-in-contextmanager
def multi_try(retries=1, wait=0):
    def _wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            for _ in range(retries + 1):  # Loop retries + first attempt
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    e = exc
                    if wait > 0:
                        time.sleep(wait)
                    pass
                    # print("fct call failed {}".format(e.args))

            raise Exception("all fct calls failed {}".format(e.args))
        return wrapped
    return _wrapper


@multi_try(20)
def connect_add_message(uuid, list_key, msg):
    print("Init redis")
    red = R()

    print("Set redis {} - {}".format(uuid, msg))
    # red.r.set(uuid, msg)
    red.r.lpush(list_key, uuid)

    # kill the connection pool after writing the message
    red.r.connection_pool.disconnect()


@multi_try(20, wait=1)
def fork():
    pid = os.fork()
    return pid


def access_shared_queue(i):
    # fork immediately
    pid = fork()
    # give each fork a new seed
    seed()

    if pid:
        uuid = get_uuid()
        # add our message to redis according to thread
        connect_add_message(uuid, "master", "hello master doofus {}".format(i))
        print("Waiting on master")

        # can be release with uuid on DB1
        RLock().add_lock_and_wait(uuid)

        print("closing master {}".format(i))
        connect_add_message(uuid + "_fin", "control", "fin master doofus {}".format(i))

        os._exit(0)

    else:
        uuid = get_uuid()
        # attempt to get a connection up to 20 times
        connect_add_message(uuid, "child", "hello child doofus {}".format(i))
        print("closing child {}".format(i))
        os._exit(0)


class R():
    def __init__(self, db=0):
        self.r = redis.StrictRedis(host='localhost', port=6379, db=db, socket_timeout=100)


class RLock(R):

    def __init__(self):
        # all locks on db1
        super(RLock, self).__init__(db=1)

    def add_lock_and_wait(self, lock_name, retry_interval=.1):
        self.r.set(lock_name, 0)

        # sleep until wake up!
        while int(self.r.get(lock_name).decode()) == 0:
            # print("sleep waiting {}".format(int(self.r.get(lock_name).decode())))
            time.sleep(retry_interval)

    # perhaps we want to release with a different message?
    def release_lock(self, lock_name):
        self.r.set(lock_name, 1)


def get_all_messages(rr, list_key):
    mc = rr.llen(list_key)
    am = list(map(lambda x: x.decode(), rr.lrange(list_key, 0, mc)))
    return mc, am


@multi_try(10, wait=.1)
def try_start_process(*args, **kwargs):
    p = Process(*args, **kwargs)
    p.start()
    return p


def main(*args, **kwargs):
    mp.set_start_method('fork')

    red = R()
    rr = red.r
    rr.flushdb()
    # create num_threads == 200
    # rr.incr("num_threads", 200)
    call_count = 50
    # m = Manager()
    # bd = m.BoundedSemaphore(min(500, call_count))
    fork_size = 15
    start_ix = 0
    all_chunks = list(np.arange(0, call_count, fork_size)[1:]) + [call_count]

    final_close = []
    for end_ix in all_chunks:

        # launch all of our processes
        all_processes = [try_start_process(target=access_shared_queue, args=[i])
                         for i in range(start_ix, end_ix)]

        # list(map(lambda x: x.start(), all_processes))
        final_close += all_processes

        while rr.llen('master') < end_ix or rr.llen('child') < end_ix:
            pass

        start_ix = end_ix

    print("Summoned Initial Messages")
    ml, master_keys = get_all_messages(rr, 'master')
    cl, child_keys = get_all_messages(rr, 'child')

    #
    # we're going to selectively continue execution for a few forks
    print("First 10 master messages {}".format(master_keys[0:10]))

    # remove the db of these messages
    rr.flushdb()

    sub_keys = np.arange(ml)
    np.random.shuffle(sub_keys)
    sub_keys = sub_keys[:int(ml/2)]
    rl = RLock()
    [rl.release_lock(master_keys[sk]) for sk in sub_keys]

    while rr.llen('control') < len(sub_keys):
        pass

    mc, control_keys = get_all_messages(rr, 'control')
    print("Finished release, subset 10 final messages {}".format(control_keys[0:10]))

    # finish the remainder (any that we didn't already clear)
    [rl.release_lock(master_keys[sk])
     for sk in range(ml)
     if sk not in sub_keys]

    list(map(lambda x: x.join(), all_processes))
    print("All processes forked, caught, released and killed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
