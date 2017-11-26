from redis import StrictRedis
from pickle import loads, dumps
from uuid import uuid4
from time import sleep
from functools import wraps


def get_uuid():
    return uuid4().hex


def get_all_messages(rr, list_key):
    mc = rr.llen(list_key)
    am = list(map(lambda x: x.decode(), rr.lrange(list_key, 0, mc)))
    return mc, am


def map_decode(l_values):
    return list(map(lambda x: x.decode(), l_values))


# https://stackoverflow.com/questions/36295766/using-yield-twice-in-contextmanager
def multi_try(retries=1, wait=0):
    def _wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            for _ in range(retries + 1):  # Loop retries + first attempt
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    e = exc
                    if wait > 0:
                        sleep(wait)
                    pass
                    # print("fct call failed {}".format(e.args))

            raise Exception("all fct calls failed {}".format(e.args))
        return wrapped
    return _wrapper


class RTraces():

    def __init__(self, db=0, *args, **kwargs):
        self.r = StrictRedis(host='localhost', db=db, port=6379, *args, **kwargs)

    def kill_connection(self):
        self.r.connection_pool.disconnect()
        return self

    @staticmethod
    def get_trace_key(trace_uuid, site_name):
        return "{}-{}".format(trace_uuid, site_name)

    @staticmethod
    def get_trace_from_key(trace_uuid):
        return trace_uuid.split("-")[0]

    # expensive method to return everything from db
    def get_all_keys(self, match="*"):
        return list(self.r.scan_iter(match=match))

    def get_all_items(self, match="*", all_keys=None):
        all_keys = self.get_all_keys(match=match) if all_keys is None else all_keys
        # queue all our requests as one
        # https://stackoverflow.com/questions/22210671/redis-python-setting-multiple-key-values-in-one-operation
        pipe = self.r.pipeline()
        for k in all_keys:
            pipe.get(k)
        return dict(zip(all_keys, pipe.execute()))

    def get_value(self, key):
        return self.r.get(key)

    def set_trace(self, trace_uuid, site_name, trace_str):
        self.r.set(RTraces.get_trace_key(trace_uuid, site_name), trace_str)
        return self

    def _clear_db(self):
        return self.r.flushdb()

    def delete_list(self, del_list):
        pipe = self.r.pipeline()
        for k in del_list:
            pipe.delete(k)
        pipe.execute()
        return self


class RControl(RTraces):

    def __init__(self):
        # all control on db2
        super(RControl, self).__init__(db=2)

    def get_control_msg(self, uuid):
        ctrl = self.r.get(uuid).decode()
        return loads(ctrl)

    def set_control_msg(self, uuid, ctrl_json):
        self.r.set(uuid, dumps(ctrl_json))
        return self


class RPairs(RTraces):

    def __init__(self):
        # all control on db2
        super(RPairs, self).__init__(db=3)

    def get_pair_uuids(self, uuid_base):
        return list(self.r.scan_iter("{}*".format(uuid_base)))
        # pair_count = self.r.llen(uuid_base)
        # # slice out these pairs, convert them to strings
        # return map_decode(self.r.lrange(uuid_base, 0, pair_count))

    # pair the uuid_base with some of the clones
    def add_pair_uuids(self, pair_uuid, value_str=b''):
        # self.r.lpush(uuid_base, uuid_clone)
        self.r.set(pair_uuid, value_str)
        return self

    @staticmethod
    def get_pair_name(uuid_base, uuid_clone, site_name):
        return "{},{}-{}".format(uuid_base, uuid_clone, site_name)

    @staticmethod
    def get_pair_from_key(trace_uuid):
        base_to_new = trace_uuid.split("-")[0]
        return base_to_new.split(",")


# support pushing multiple locks. if you wait on a lock that already has a key, you'll immediately pull
class RLock(RTraces):

    def __init__(self):
        # all locks on db1
        super(RLock, self).__init__(db=1)

    def add_lock_and_wait(self, lock_name, retry_interval=.1):

        # check if a lock exists
        lock_exists = self.r.exists(lock_name) and self.r.llen(lock_name) > 0

        # sleep until wake up!
        while not lock_exists:
            # check for existence
            lock_exists = self.r.exists(lock_name) and self.r.llen(lock_name) > 0

            # print("sleep waiting {}".format(int(self.r.get(lock_name).decode())))
            sleep(retry_interval)

        # otherwise, we already have a lock to process
        # pull from the front of the list (oldest keys, fifo)
        r_val = loads(self.r.lpop(lock_name))

        # then return
        return r_val

    # perhaps we want to release with a different message?
    def release_lock(self, lock_name, command_behavior):
        # push the end of the list (right push)
        self.r.rpush(lock_name, dumps(command_behavior))
        return self


class RMessages(RTraces):

    def __init__(self):
        # all control on db2
        super(RMessages, self).__init__(db=4)

    def set_msg(self, uuid_base, msg):
        self.r.set(uuid_base, msg)
        return self

    def scan_messages_iter(self, uuid_base):
        return self.r.scan_iter(uuid_base)
