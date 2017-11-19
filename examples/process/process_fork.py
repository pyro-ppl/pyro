from pdb import set_trace as bb
from threading import Thread, Semaphore
import sys
import time
import os
from multiprocessing import Process, Manager, Semaphore, Lock, Queue, Array
# from arrow import get as aget
from arrow import now as anow
from uuid import uuid4 as uuid
from numpy.random import normal
# from pipeproxy import proxy

def nts():
    return anow().timestamp


def sample(mu, sigma):
    return normal(mu, sigma)


def store_fork_continue(fct_key, mu_sigma, trace_dict, shared_dict, shared_queue, lock, manager, kill_main=False):
    # store our new semaphore
    # print("entering store")
    # print("Forking {}:, in semdict {}".format(fct_key, os.getpid()))
    shared_semaphore = manager.Semaphore(0)
    pid = os.fork()
    local_uuid = uuid().hex
    # mark this spot with a uuid
    trace_dict[fct_key] = {'uuid': local_uuid}

    # child
    if pid == 0:
        # frozen
        # print("Child {} waiting {}".format(fct_key, os.getpid()))
        shared_semaphore.acquire()
        # with lock:

        print("grabbing queue")
        # gg = shared_dict.get()
        gg = shared_dict[local_uuid]
        print(gg)
        # print("Sahred array: {}".format(shared_queue))
        # print("Shared info: {}".format(shared_dict.get(local_uuid)))

        # print("Child {} released {}".format(fct_key, os.getpid()))
        store_fork_continue(fct_key, sample(*mu_sigma), trace_dict, shared_dict, shared_queue,
                            lock, manager, kill_main=True)

    # if not child, we continue on the way
    else:
        # mark the trace with a unique
        # uuid referncing this exact location
        trace_dict[fct_key]['thread'] = shared_semaphore
        trace_dict[fct_key]['value'] = sample(*mu_sigma)

        with lock:
            shared_dict.update({local_uuid: dict(**trace_dict)})
        # shared_queue.put({local_uuid: dict(**trace_dict)})
        # # append to shared dict of traces
        # with shared_dict.get_lock():
        #     # create a shared dictionary
        #     print("Accessing semaphore")
        #     shared_dict.update({local_uuid:  dict(**trace_dict)})
        #     print("Donezo with semaphore")

        if kill_main:
            print("Killing main forked thread")
            os._exit()

        # print("parent continues")
        # shared_dict[key]['current'] = shared_semaphore
        # print("Fork {} continuing, in semdict {}".format(key, shared_dict[key].keys()))


def l_trace_dict(td):
    return list(map(lambda x: "{}: {}".format(x[0], x[1]['value']), td.items()))


def model_sim(queue, shared_dict, shared_array, lock, manager):

    trace_dict = {}

    print("Running msim more thing on master")
    store_fork_continue("other", (0, 1),
                        trace_dict, shared_dict, queue,
                        lock, manager)
    return trace_dict

    # def oh_and_one_more_thing():
    # print("td oaomt: {}".format(trace_dict))

    # def do_a_thing():
    #     store_fork_continue("basic", (0, 1), trace_dict,
    #                         shared_dict, lock, manager)
    #     # print("td dtat: {}".format(trace_dict))
    #     oh_and_one_more_thing()

# class Example:
#     def __init__(self):
#         self.parameter = None

#     def setParameter(self, parameter):
#         print "setting parameter to: " + str(parameter)
#         self.parameter = parameter

#     def getParameter(self):
#         print "getting parameter: " + str(self.parameter)
#         return self.parameter


def main():

    # share a reference to the same semaphore
    manager = Manager()
    shared_dict = manager.dict()
    lock = Lock()
    queue = Queue()
    shared_array = manager.list()

    # example = dict({})
    # shared_dict, exampleProxyListener = proxy.createProxy(example)
    # exampleProxyListener.listen()

    # ss = manager.Semaphore(0)
    # bb()
    # def freeze(pid):
    #     sem_dict[pid] = manager.Semaphore(0)
    #     return sem_dict[pid]
    trace_dict = {}

    master_pid = os.getpid()
    print("TOP: original pid {}".format(master_pid))

    # def oh_and_one_more_thing():
    #     print("Running one more thing on master")
    #     store_fork_continue("other", (0, 1),
    #                         trace_dict, shared_dict, lock, manager)
    #     # print("td oaomt: {}".format(trace_dict))

    # def do_a_thing():
    #     store_fork_continue("basic", (0, 1), trace_dict,
    #                         shared_dict, lock, manager)
    #     # print("td dtat: {}".format(trace_dict))
    #     oh_and_one_more_thing()

    # get the original pid
    # do_a_thing()
    def print_sem():
        print("queue print")
        print("{}".format(queue.get()))
        # with shared_dict.get_lock():
            # print("Semaphore print")
            # print("{}".format(shared_dict.keys()))

    if os.getpid() == master_pid:
        print("MASTER: Finished, entering loop")
        print("o trace: {}".format(l_trace_dict(trace_dict)))

        reply = 'c'
        while reply != 'e':
            if reply == 'b':
                bb()
            #
            if reply == 'p':
                print_sem()

            if reply == 'i':
                td = model_sim(queue, shared_dict, shared_array, lock, manager)
                print("trace {}".format(td))
                # oh_and_one_more_thing()
                # print_sem()
            if reply == 'q':
                print("fetching queue")
                qi = queue.get()
                print("getting uuid")
                uuid_key = list(qi.keys())[0]
                print("release the kraken!")
                print(qi)
                qi[uuid_key]["other"]['thread'].release()
                # shared_array.append({uuid_key: 5})
                # shared_dict.update({uuid_key: 5})
                print(shared_dict)

            if reply == 'f':
                print("MASTER: attempting basic semaphore call")
                trace_dict["basic"]['thread'].release()

            if reply == 'o':
                print("MASTER: attempting other call")
                trace_dict["other"]['thread'].release()

            # print("accessing sem_dict")
            # print("sd keys {}".format(sem_dict.keys()))
            reply = input("f for function call / e for exit\n")

        os._exit(0)
    else:
        # print("sd keys {}".format(sem_dict.keys()))
        print("Watch out, forked child coming through {}".format(os.getpid()))
        print("td {}".format(l_trace_dict(trace_dict)))
        os._exit(0)

    # for _ in range(3):
    #     sem_dict["basic"].release()

    # print("TOP: original pid {}".format(pid))
    # frozen = freeze(pid)

    # # create our semaphore and freeze it
    # orig_pid = os.fork()

    # if orig_pid == 0:
    #     frozen.acquire()
    #     this_pid = os.getpid()
    #     freeze(this_pid)
    #     # ns_pid = os.fork()

    #     print("CHILD: unfrozen")
    # else:
    #     print("PARENT: Original {}, fork: {}".format(pid, orig_pid))
    #     frozen.release()
    #     print("PARENT: completed")
    #     frozen.release()

    # if orig_pid != 0:
    #     sem_dict["master"] = orig_pid

    # for i in range(2):
    #     pid = os.fork()
    #     sem_dict[pid] = True

    # print(pid)

    # if orig_pid != 0 and orig_pid not in sem_dict:
    #     sem_dict[orig_pid] = True
    #     print(sem_dict.keys())

    # if pid != 0:
    #     sys.exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('-cfg', "--config_file", type=str, required=True,
    # help='master configuration file for running the training experiments')
    # parser.add_argument('-tgt', "--target_file", type=str, required=True,
    #                     help='target to match')

    # parser.add_argument('-iw', "--initial_width", type=float,
    #                     help='width of lines to draw')

    args = parser.parse_args()

    # turn args into a list of kwargs sent to main
    main(**vars(args))
    # try:
    # except AssertionError:
    #     pass
    # except EOFError:
    #     pass
