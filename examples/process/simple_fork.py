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
from numpy.random import normal


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

def foo(shared_dict):
    local_uuid = get_uuid()
    shared_dict[local_uuid] = sample(0,1)


def main(*args, **kwargs):
    mp.set_start_method('fork')
    manager = Manager()
    shared_obj = manager.dict()

    def run_and_kill():
        p = mp.Process(target=foo, args=(shared_obj,))
        p.start()
        p.join()


    reply = 'c'
    while reply != 'e':

        if reply == 'b':
            bb()

        if reply == 'f':
            print("running model fct")
            run_and_kill()
            print(repr(shared_obj))

        reply = input("f for function call / e for exit\n")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))