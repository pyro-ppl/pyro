Reparameterizers
================
.. automodule:: pyro.infer.reparam

The :mod:`pyro.infer.reparam` module contains reparameterization strategies for
the :func:`pyro.poutine.handlers.reparam` effect. These are useful for altering
geometry of a poorly-conditioned parameter space to make the posterior better
shaped. These are useful in e.g. ``Auto*Normal`` guides and MCMCM.

Loc-Scale Decentering
---------------------
.. automodule:: pyro.infer.reparam.loc_scale
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__

Transform
---------
.. automodule:: pyro.infer.reparam.transform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__

Neural Transport
----------------
.. automodule:: pyro.infer.reparam.neutra
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__

Cumsum
------
.. automodule:: pyro.infer.reparam.cumsum
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__

Levy Stable
-----------
.. automodule:: pyro.infer.reparam.stable
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__

Levy-Ito Decompositions
-----------------------
.. automodule:: pyro.infer.reparam.levy_ito
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__
