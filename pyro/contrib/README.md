# Contributed Code

Code in `pyro.contrib` is under various stages of development.
This code makes no guarantee about maintaining backwards compatibility.

To add a new contrib module `foo`:
1. Create a module `pyro/contrib/foo.py` or `pyro/contrib/foo/__init__.py` with new code.
2. Create a module docstring describing purpose and authors.
3. Create document all public functions and methods.
4. Create Sphinx hooks in `docs/source/contrib.rst` and `docs/source/contrib.foo.rst`.
5. Create tests `tests/contrib/test_foo.py` or `tests/contrib/foo/test_*.py`.
6. Create examples `examples/contrib/foo/*.py`.
