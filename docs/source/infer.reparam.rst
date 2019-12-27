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
    :member-order: bysource

Transform
---------
.. automodule:: pyro.infer.reparam.transform
    :members:
    :member-order: bysource

Neural Transport
----------------
.. automodule:: pyro.infer.reparam.neutra
    :members:
    :member-order: bysource

Stable
------
.. automodule:: pyro.infer.reparam.stable
    :members:
    :member-order: bysource

Levy-Ito Decomposition
----------------------
.. automodule:: pyro.infer.reparam.levy_ito
    :members:
    :member-order: bysource
