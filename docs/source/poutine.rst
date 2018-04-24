Poutine (Effect handlers)
==========================

Beneath the built-in inference algorithms, Pyro has a library of composable
effect handlers for creating new inference algorithms and working with probabilistic
programs. Pyro's inference algorithms are all built by applying these handlers to stochastic functions.

Handlers
---------

.. automodule:: pyro.poutine.handlers
    :members:
    :undoc-members:
    :show-inheritance:

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
