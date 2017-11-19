from pdb import set_trace as bb
from threading import Thread, Semaphore
import sys
import time
import os
import multiprocessing as mp
from multiprocessing import Process, Manager, Semaphore, Lock, Queue, Array
# from arrow import get as aget
from arrow import now as anow
from uuid import uuid4
from numpy.random import normal, seed


def get_uuid():
    return uuid4().hex


def sample(mu, sigma):
    return normal(mu, sigma)

def store_fork_continue(*args, **kwargs):
    manager = kwargs['manager']
    trace_dict = kwargs['trace_dict']
    fct_key = kwargs['fct_key']
    mu_sigma = kwargs['fct_args']

    shared_semaphore = manager.Semaphore(0)

    if fct_key not in trace_dict and local_uuid not in trace_dict[fct_key]:
        local_uuid = get_uuid()
        # mark this spot with a uuid
        trace_dict[fct_key] = {'uuid': local_uuid}

    # child of the fork
    if pid == 0:
        shared_semaphore.acquire()
        # run it again
        p = mp.Process(target=store_fork_continue, args=args, kwargs=kwargs)
        p.start()
        # store_fork_continue(*args, **kwargs)

    # if not child, we continue on the way
    else:
        trace_dict[fct_key]['thread'] = shared_semaphore
        trace_dict[fct_key]['value'] = sample(*mu_sigma)


def foo(local_uuid, lock, shared_dict, manager, main_exit=False):
    print("Fork for your life {}".format(os.getpid()))
    if local_uuid not in shared_dict:
        shared_dict[local_uuid] = {'value': 0, 'thread': None}
        # : manager.list()}

    pid = os.fork()
    seed()
    print("Post forking {}".format(pid))
    # master
    if pid:

        print("Main so hot post pid right now")
        with lock:
            try:
                print("in lock")
                # old_val = shared_dict.pop(local_uuid, None)
                # print("old val eradticated")
                shared_dict.update({local_uuid: {'value': sample(0, 1)}})
                print("val changed")
                # shared_dict[local_uuid]['thread'] = shared_semaphore
                # shared_dict.update({local_uuid: {'value': sample(0, 1), 'thread': shared_semaphore}})
                print("shared dict updated")
            except Exception as e:
                print("Issue with update {}".format(e))
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
        shared_semaphore = manager.Semaphore(0)

        shared_dict.update({local_uuid + "_thread": {'thread': shared_semaphore}})

        # time.sleep(0)
        shared_semaphore.acquire()
        # print("Child fork, dying after release")
        # os._exit(0)
        print("Child fork, forking after release")
        # # call in again plz -- why stop the fun?
        foo(local_uuid, lock, shared_dict, manager, main_exit=True)

        print("Post foo existance")
        os._exit(0)



def main(*args, **kwargs):
    mp.set_start_method('fork')
    manager = Manager()
    shared_obj = manager.dict()
    all_processes = []
    lock = Lock()


    def run_and_kill():
        p = mp.Process(target=foo, args=(get_uuid(), lock, shared_obj, manager,))
        p.start()
        all_processes.append(p)
        # p.join()


    reply = 'c'
    while reply != 'e':

        if reply == 'b':
            bb()

        if reply == 'f':
            print("running model fct")
            run_and_kill()
            print(repr(shared_obj))

        reply = input("f for function call / e for exit\n")


    # for i in range(10):
    #     print("rnc {}".format(i))
    #     run_and_kill()

    # time.sleep(3)
    for i in range(10):
        obj_threads = [key for key in shared_obj.keys() if '_thread' in key]
        o1 = obj_threads[0]
        print('sh : {}'.format(shared_obj[o1.replace('_thread', '')]))
        time.sleep(1)
        shared_obj[o1]['thread'].release()

    time.sleep(1)
    print("wait all")
    for mz in all_processes:
        mz.join()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))