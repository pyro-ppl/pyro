Optimization
============

The module `pyro.optim` provides support for optimization in Pyro. In particular
it provides `PyroOptim`, which is used to wrap PyTorch optimizers
and manage optimizers for dynamically generated parameters
(see the tutorial `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for
a discussion). Any custom optimization algorithms are also to be found here.

.. include:: pyro.optim.txt
