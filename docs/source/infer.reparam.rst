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

Transform
---------
.. automodule:: pyro.infer.reparam.transform
    :members:

Neural Transport
----------------
.. automodule:: pyro.infer.reparam.neutra
    :members:

Cumsum
------
.. automodule:: pyro.infer.reparam.cumsum
    :members:

Levy Stable
-----------
.. automodule:: pyro.infer.reparam.stable
    :members:

Levy-Ito Decompositions
-----------------------
.. automodule:: pyro.infer.reparam.levy_ito
    :members:
