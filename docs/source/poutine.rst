Poutine (Effect handlers)
==========================

Beneath the built-in inference algorithms, Pyro has a library of composable
effect handlers for creating new inference algorithms and working with probabilistic
programs. Pyro's inference algorithms are all built by applying these handlers to stochastic functions.

Handlers
---------

.. autofunction:: pyro.poutine.block

.. autofunction:: pyro.poutine.condition

.. autofunction:: pyro.poutine.do

.. autofunction:: pyro.poutine.enum

.. autofunction:: pyro.poutine.escape

.. autofunction:: pyro.poutine.indep

.. autofunction:: pyro.poutine.infer_config

.. autofunction:: pyro.poutine.lift

.. autofunction:: pyro.poutine.replay

.. autofunction:: pyro.poutine.queue

.. autofunction:: pyro.poutine.scale

.. autofunction:: pyro.poutine.trace

Trace
------

.. autoclass:: pyro.poutine.trace_struct.Trace
    :members:
    :undoc-members:
    :show-inheritance:

Messengers
-----------

.. include:: pyro.poutine.txt

Runtime
--------

.. automodule:: pyro.poutine.runtime
    :members:
    :undoc-members:
    :show-inheritance:
