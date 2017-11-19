import optparse
import time
from pdb import set_trace as bb

import greenstack as greenlet
from greenstack import greenstack as Greenlet


def link(next_greenlet):
    value = greenlet.getcurrent().parent.switch()
    print("cur : {} , par: {}, val: {}".format(greenlet.getcurrent(), greenlet.getcurrent().parent, value))
    next_greenlet.switch(value + 1)


def chain(n):
    start_node = greenlet.getcurrent()
    for i in range(n):
        g = Greenlet(link)
        g.switch(start_node)
        start_node = g

    bb()
    return start_node.switch(0)

if __name__ == '__main__':
    p = optparse.OptionParser(
        usage='%prog [-n NUM_COROUTINES]', description=__doc__)
    p.add_option(
        '-n', type='int', dest='num_greenlets', default=100000,
        help='The number of greenlets in the chain.')
    options, args = p.parse_args()

    if len(args) != 0:
        p.error('unexpected arguments: %s' % ', '.join(args))

    start_time = time.clock()
    print('Result:', chain(options.num_greenlets))
    print(time.clock() - start_time, 'seconds')