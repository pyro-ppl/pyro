from pdb import set_trace as bb
from threading import Thread, Semaphore
import sys
import time
import os
import multiprocessing as mp
from multiprocessing import Process, Manager, Semaphore, Lock, Queue, Array
from multiprocessing.managers import BaseManager, SyncManager, ListProxy
import numpy as np
# from arrow import get as aget
from arrow import now as anow
from uuid import uuid4
from numpy.random import normal, seed, choice
from collections import defaultdict
import contextlib
import functools

# Forking multiprocessing:
# https://github.com/python/cpython/blob/3972628de3d569c88451a2a176a1c94d8822b8a6/Lib/multiprocessing/popen_fork.py

# Managers:
# https://github.com/python/cpython/blob/3972628de3d569c88451a2a176a1c94d8822b8a6/Lib/multiprocessing/context.py



def get_uuid():
    return uuid4().hex


def sample(mu, sigma):
    return normal(mu, sigma)



def fork(fct_key, local_uuid, trace_dict, shared_dict, thread_dict, manager,
         clone_op=False, clone_thread=None):
    print("Fork for your life {}".format(os.getpid()))
    pid = os.fork()
    seed()
    print("Post forking {}".format(pid))
    # master
    if pid:

        print("Main so hot post pid right now")
        if not clone_op:
            # with lock:
            #     try:
            print("in lock")
            # sample and store to the function key
            trace_dict[fct_key] = sample(0, 1)
            # old_val = shared_dict.pop(local_uuid, None)
            # print("old val eradticated")
            shared_dict.update({local_uuid: trace_dict})
            print("val changed")
            # shared_dict[local_uuid]['thread'] = shared_semaphore
            # shared_dict.update({local_uuid: {'value': sample(0, 1), 'thread': shared_semaphore}})
            print("shared dict updated")
            # except Exception as e:
            #     print("Issue with update {}".format(e))
        # print("Main coming through: {}".format(shared_dict[local_uuid]))

            # print("Waiting on child")
            print("Finished waitin, killing main")
            # os.waitpid(pid, 0)
            # print("Finished waitin, doing sumtin")
            os._exit(0)
        # if main_exit:
        #     print("waiting then killing main {}".format(pid))
        #     os._exit(0)

    else:
        print("semaphore purgatory")
        if not clone_op:
            shared_semaphore = manager.Semaphore(0)

            # thread_list = manager.list()
            # thread_list.append(shared_semaphore)

            # want to continue from here?
            thread_dict.update({local_uuid: shared_semaphore})

            # time.sleep(0)
            time.sleep(1)
            shared_semaphore.acquire()
            clone_count = thread_dict[local_uuid]

            for i in range(clone_count):
                print("Cloning {}_{}".format(local_uuid, i))
                fork(fct_key, local_uuid, trace_dict, shared_dict, thread_dict, manager,
                     clone_op=True, clone_thread=i)

            print("Killing the clone loop")
            os._exit(0)

        else:
            clone_id = "{}_{}".format(local_uuid, clone_thread)
            print("In cloning op {}".format(clone_id))

            # when you're the clone operation,
            shared_semaphore = manager.Semaphore(0)
            print("updating thread dict {} - {}".format(clone_id, len(shared_dict)))
            thread_dict.update({clone_id: shared_semaphore})
            # thread_dict[local_uuid]['threads'].append(shared_semaphore)
            # thread_dict.update({clone_id: {'thread': shared_semaphore}})

            print("HACK to finish the update across the wire-- this needs to be fixed")
            time.sleep(1)

            # while True:
            # print("finished thread dict {} - {}".format(clone_id, len(shared_dict)))
            print("finished 2thread dict {} - {}".format(clone_id, len(shared_dict)))
            # gg = thread_dict[local_uuid]
            # print("TH type {}".format(thread_list))
            # with thread_list:
            # gg['threads'].append(shared_semaphore)
            # clone_thread.append(shared_semaphore)
            # thread_dict.update({local_uuid: gg})
            shared_semaphore.acquire()

            print("HACK to finish the update post release")
            time.sleep(1)
            # print("Child fork, dying after release")
            # os._exit(0)
            print("Cloned accordingly, then wait for the fork")
            # # call in again plz -- why stop the fun?
            fork(fct_key, local_uuid, trace_dict, shared_dict, thread_dict, manager)

            print("Post fork existance")
            os._exit(0)


def model(shared_traces, shared_threads, manager):

    # go until you hit the first
    # faux tracing
    trace_dict = {}

    uuid = get_uuid()

    # call fork for our first "sample"
    fork("first", uuid, trace_dict, shared_traces, shared_threads, manager)


def register_shared_objects(cls, keys_and_callables):
    for key, kw in keys_and_callables.items():
        print("registering {} - {}".format(key, kw))
        cls.register(key, **kw)

def connect_retry(max_tries):
    while max_tries > 0:
        try:
            dm = DM()
            dm.connect()
            return dm
        except Exception:
            print("Failed connect")  # .format(e))
            max_tries -= 1

    return None


def try_get(max_tries, fct, *args, **kwargs):
    # try everythign
    dm = connect_retry(max_tries)

    while max_tries > 0:
        try:
            return dm, getattr(dm, fct)(*args, **kwargs)
        except Exception:
            print("Failed try {}".format(fct))
            max_tries -= 1


# https://stackoverflow.com/questions/36295766/using-yield-twice-in-contextmanager
def multi_try(retries=1):
    def _wrapper(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            for _ in range(retries + 1):  # Loop retries + first attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print("fct call failed {}".format(e.args))
        return wrapped
    return _wrapper


@multi_try(20)
def connect_add_message(msg):

    msg = "/".join([msg for i in range(25)])
    random_list = np.random.randint(10)
    dm = DM(random_list)
    print("Connect")
    dm.connect()
    print("Access")

    print("update {} accessing {}".format(msg, random_list))
    # dm.get_shared_queue().put({get_uuid(): msg})
    # dm.get_shared_globals().update({get_uuid(): msg})
    # shared_list = dm.get_shared_globals().get('shared_list_{}'.format(random_list))
    shared_list = dm.get_shared_globals().get('shared_list')
    # print("update {}".format(msg))
    # time.sleep(.1 + np.random.uniform(3))
    shared_list.append(msg)
    print("finish {}".format(msg))


def access_shared_queue(i):
    # pre-fork we add registers, they'll be there post fork
    add_registers()
    pid = os.fork()
    seed()
    if pid:
        # attempt to get a connection up to ten times
        connect_add_message("hello master doofus {}".format(i))
        print("closing master {}".format(i))
        time.sleep(3)
        os._exit(0)

    else:

        connect_add_message("hello child doofus {}".format(i))
        print("closing child {}".format(i))
        time.sleep(3)
        os._exit(0)


class DM(SyncManager):
    def __init__(self, rand_val=0):
        super(DM, self).__init__(address=('127.0.0.1', 50000 + rand_val))  # , authkey=b'abracadabra')

def add_registers():
    DM.register("get_shared_queue")
    DM.register("get_shared_globals")
    DM.register("create_list")
    DM.register("get_shared_traces")
    DM.register("get_shared_threads")

def start_dm(i):
    shared_queue = Queue()
    traces = {}
    threads = {}
    shared_globals = {}
    # token_list = []
    # list_proxy = ListProxy(token_list)
    # shared_lists = defaultdict(list)

    # DM.register('get_list', list, proxytype=ListProxy)
    DM.register('get_shared_queue', callable=lambda: shared_queue)
    DM.register('get_shared_globals', callable=lambda: shared_globals)
    DM.register('create_list', list, proxytype=ListProxy)
    DM.register('get_shared_traces', callable=lambda: traces)
    DM.register('get_shared_threads', callable=lambda: threads)
    DM(i).get_server().serve_forever()


@multi_try(5)
def mt():
    print("Attempting")
    raise Exception("trying this thing out")


def random_delay_target(fct, *args, **kwargs):
    # seed()
    # time.sleep(.01 + np.random.uniform(0, 2))
    fct(*args, **kwargs)


def main(*args, **kwargs):
    mp.set_start_method('fork')

    all_processes = []
    sp = 10
    for i in range(sp):
        s = Process(target=start_dm, args=[i])
        s.daemon = True
        s.start()

    time.sleep(.1)
    add_registers()

    mvs = []
    for i in range(sp):
        m = DM(i)
        m.connect()

        # round_robin_list = {'shared_list_{}'.format(i): m.list()
        #                     for i in range(50)}

        # m.get_shared_globals().update(round_robin_list)  # {'shared_list': gl})
        gl = m.list()
        m.get_shared_globals().update({'shared_list': gl})
        mvs.append(m)
    # glist = m.create_list()
    # gg = )
    # .update({'shared_list': glist})
    time.sleep(.5)

    call_count = 100
    all_processes += [Process(target=random_delay_target, args=[access_shared_queue, i])
                      for i in range(call_count)]

    list(map(lambda x: x.start(), all_processes))
    time.sleep(2.5)
    bb()
    # print("Sleep waiting")
    all_messages = 0
    all_list = []
    for m in mvs:
        # for mname in round_robin_list:
        mname = 'shared_list'
        mlist = list(m.get_shared_globals().get(mname))
        all_messages += len(mlist)
        all_list.extend(mlist)

    am = all_messages
    al = all_list
    # mlist = m.get_shared_globals().get('shared_list')
    # messages = [mlist[i] for i in range(len(mlist))]
    bb()
    list(map(lambda x: x.join(), all_processes))
    # messages = [m.get_shared_queue().get() for i in range(2*call_count)]
    bb()
    s.join()

    # proc_address = ('127.0.0.1', 2000)

    # print("Begin shared server")
    # start_server(DopeManager, proc_address)
    # # s = Process(target=start_server, args=(DopeManager, proc_address,))
    # # s.daemon = True
    # # s.start()

    # # print("Letting server start, then access shared queue")
    # time.sleep(2)
    # fake_args = {}
    # # DopeManager.register("nothing", callable=lambda: fake_args)
    # dm = DopeManager(address=proc_address, authkey=b'dope')
    # dm.connect()
    # bb()
    # # now access shared queue
    # print("Begin acccess shared queue")
    # m = Process(target=access_shared_queue, args=(DopeManager, proc_address,))
    # # # runnign
    # m.start()

    # # # running
    # m.join()
    # # add_server_shared(DopeManager)



    bb()
    # manager = Manager()

    # shared_traces = manager.dict()
    # shared_threads = manager.dict()


    # shared_obj = manager.dict()
    all_processes = []


    def run_and_kill():
        p = mp.Process(target=model, args=(shared_traces, shared_threads, manager,))
        p.start()
        all_processes.append(p)
        # p.join()


    num_particles = 10
    shared_traces.clear()
    shared_threads.clear()

    for i in range(num_particles):
        run_and_kill()

    while len(shared_traces.keys()) < num_particles or \
            len(shared_threads.keys()) < num_particles:
        # print("Waiting on traces")
        pass

    all_keys = list(shared_threads.keys())
    all_threads = dict(**shared_threads)
    num_clones = 20
    clone_counts = defaultdict(lambda: 0)
    for nn in range(num_clones):
        sel = choice(all_keys)

        # count the number of forks
        clone_counts[sel] += 1

    # got our previous
    prev_shared = len(shared_threads)

    # update with new info
    shared_threads.update(dict(**clone_counts))

    # now go through and call release for the previous threads
    for key in clone_counts:
        all_threads[key].release()

    print("Waiting on copies {} - {}: clones {}".format(
                                                len(shared_threads),
                                                prev_shared,
                                                num_clones))
    # wait for the clone
    while len(shared_threads) - prev_shared < num_clones:
        pass



    bb()


    reply = 'c'
    while reply != 'e':

        if reply == 'b':
            bb()

        if reply == 'f':
            print("running model fct")
            run_and_kill()
            # print(repr(shared_obj))

        reply = input("f for function call / e for exit\n")


    # for i in range(10):
    #     print("rnc {}".format(i))
    #     run_and_kill()

    # time.sleep(3)
    # for i in range(10):
    #     obj_threads = [key for key in shared_obj.keys() if '_thread' in key]
    #     o1 = obj_threads[0]
    #     print('sh : {}'.format(shared_obj[o1.replace('_thread', '')]))
    #     time.sleep(1)
    #     shared_obj[o1]['thread'].release()

    # time.sleep(1)
    print("wait all")
    for mz in all_processes:
        mz.join()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))