Installation
============

Install from Source
-------------------
Pyro supports Python 2.7.* and Python 3.  To setup, install `Pytorch <http://pytorch.org>`_ then run::

   pip install pyro-ppl

or install from source::

   git clone https://github.com/uber/pyro.git
   cd pyro
   python setup.py install

.. warning::

    Some features of Pyro require bleeding-edge features of PyTorch (e.g. `enum_discrete`).
    We recommend installing PyTorch from source using the `master` branch.
    See `PyTorch install instructions <https://github.com/pytorch/pytorch#from-source>`_ for details.
