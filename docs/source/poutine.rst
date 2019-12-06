Poutine (Effect handlers)
==========================

Beneath the built-in inference algorithms, Pyro has a library of composable
effect handlers for creating new inference algorithms and working with probabilistic
programs. Pyro's inference algorithms are all built by applying these handlers to stochastic functions.

Handlers
---------

.. automodule:: pyro.poutine.handlers
    :members:

.. autofunction:: pyro.infer.enum.config_enumerate

Trace
------

.. autoclass:: pyro.poutine.Trace
    :members:
    :undoc-members:
    :show-inheritance:

Runtime
--------

.. automodule:: pyro.poutine.runtime
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
----------

.. automodule:: pyro.poutine.util
    :members:
    :undoc-members:
    :show-inheritance:

Messengers
-----------

Messenger objects contain the implementations of the effects exposed by handlers.
Advanced users may modify the implementations of messengers behind existing handlers or write new messengers
that implement new effects and compose correctly with the rest of the library.

.. include:: pyro.poutine.txt
