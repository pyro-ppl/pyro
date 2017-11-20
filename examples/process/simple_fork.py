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
from numpy.random import normal, seed, choice
from collections import defaultdict

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


def main(*args, **kwargs):
    mp.set_start_method('fork')
    manager = Manager()

    shared_traces = manager.dict()
    shared_threads = manager.dict()
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