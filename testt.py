import socket
import time
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from queue import Empty

DEVICE = "cuda"
N = 10
a = torch.tensor([1., 2., 3], device=DEVICE)

class Proc(object):
    def __init__(self, queue, event, i):
        self.queue = queue
        self.event = event
        self.i = i

    def run(self):
        t = torch.tensor(0., device=DEVICE)
        n = 0
        while n < N:
            t = t+1
            print("sent:", t + self.i)
            b = a + 2
            self.queue.put({"a": a, "b": b, "t": t + self.i})
            n += 1
            self.event.wait()
            self.event.clear()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    event = ctx.Event()
    event1 = ctx.Event()
    proc = Proc(queue, event, 0)
    proc1 = Proc(queue, event1, 10)
    p = ctx.Process(name="p0", target=proc.run)
    p.daemon = True
    p.start()
    p1 = ctx.Process(name="p1", target=proc1.run)
    p1.daemon = True
    p1.start()
    ps = [p, p1]

    for i in range(20):
        try:
            recv = queue.get(timeout=5)
            print("recvddddddddddd:", recv)
            if recv["t"] > 10:
                event1.set()
            else:
                event.set()
        except socket.error as e:
            if getattr(e, "errno", None) == errno.ENOENT:
                pass
            else:
                raise e
        except Empty:
            continue

    p.join()
    p1.join()