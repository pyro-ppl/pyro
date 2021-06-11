Biological Sequence Models with MuE
===================================
.. automodule:: pyro.contrib.mue

.. warning:: Code in ``pyro.contrib.mue`` is under development.
    This code makes no guarantee about maintaining backwards compatibility.

``pyro.contrib.mue`` provides modeling tools for working with biological
sequence data. In particular it implements MuE distributions, which are used as
a fully generative alternative to multiple sequence alignment-based
preprocessing.

Reference:
MuE models were described in Weinstein and Marks (2021),
https://www.biorxiv.org/content/10.1101/2020.07.31.231381v2.

Example MuE Models
------------------
.. automodule:: pyro.contrib.mue.models
    :members:
    :show-inheritance:
    :member-order: bysource

State Arrangers for Parameterizing MuEs
---------------------------------------
.. automodule:: pyro.contrib.mue.statearrangers
    :members:
    :show-inheritance:
    :member-order: bysource

Missing or Variable Length Data HMM
-----------------------------------
.. automodule:: pyro.contrib.mue.missingdatahmm
    :members:
    :show-inheritance:
    :member-order: bysource


Biosequence Dataset Loading
---------------------------
.. automodule:: pyro.contrib.mue.dataloaders
    :members:
    :show-inheritance:
    :member-order: bysource
