# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example usage::

    # Simple getting and setting.
    print(pyro.settings.get())  # print all settings
    print(pyro.settings.get("cholesky_relative_jitter"))  # print one
    pyro.settings.set(cholesky_relative_jitter=0.5)  # set one
    pyro.settings.set(**my_settings)  # set many

    # Use as a contextmanager.
    with pyro.settings.context(cholesky_relative_jitter=0.5):
        my_function()

    # Use as a decorator.
    fn = pyro.settings.context(cholesky_relative_jitter=0.5)(my_function)
    fn()

    # Register a new setting.
    pyro.settings.register(
        "binomial_approx_sample_thresh",  # alias
        "pyro.distributions.torch",       # module
        "Binomial.approx_sample_thresh",  # deep name
    )

    # Register a new setting on a user-provided validator.
    @pyro.settings.register(
        "binomial_approx_sample_thresh",  # alias
        "pyro.distributions.torch",       # module
        "Binomial.approx_sample_thresh",  # deep name
    )
    def validate_thresh(thresh):  # called each time setting is set
        assert isinstance(thresh, float)
        assert thresh > 0

Default Settings
----------------

{defaults}

Settings Interface
------------------
"""

# This library must have no dependencies on other pyro modules.
import functools
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

# Docs are updated by register().
_doc_template = __doc__

# Global registry mapping alias:str to (modulename, deepname, validator)
# triples where deepname may have dots to indicate e.g. class variables.
_REGISTRY: Dict[str, Tuple[str, str, Optional[Callable]]] = {}


def get(alias: Optional[str] = None) -> Any:
    """
    Gets one or all global settings.

    :param str alias: The name of a registered setting.
    :returns: The currently set value.
    """
    if alias is None:
        # Return dict of all settings.
        return {alias: get(alias) for alias in sorted(_REGISTRY)}
    # Get a single setting.
    module, deepname, validator = _REGISTRY[alias]
    value = import_module(module)
    for name in deepname.split("."):
        value = getattr(value, name)
    return value


def set(**kwargs) -> None:
    r"""
    Sets one or more settings.

    :param \*\*kwargs: alias=value pairs.
    """
    for alias, value in kwargs.items():
        module, deepname, validator = _REGISTRY[alias]
        if validator is not None:
            validator(value)
        destin = import_module(module)
        names = deepname.split(".")
        for name in names[:-1]:
            destin = getattr(destin, name)
        setattr(destin, names[-1], value)


@contextmanager
def context(**kwargs) -> Iterator[None]:
    r"""
    Context manager to temporarily override one or more settings. This also
    works as a decorator.

    :param \*\*kwargs: alias=value pairs.
    """
    old = {alias: get(alias) for alias in kwargs}
    try:
        set(**kwargs)
        yield
    finally:
        set(**old)


def register(
    alias: str,
    modulename: str,
    deepname: str,
    validator: Optional[Callable] = None,
) -> Callable:
    """
    Register a global settings.

    This should be declared in the module where the setting is defined.

    This can be used either as a declaration::

        settings.register("my_setting", __name__, "MY_SETTING")

    or as a decorator on a user-defined validator function::

        @settings.register("my_setting", __name__, "MY_SETTING")
        def _validate_my_setting(value):
            assert isinstance(value, float)
            assert 0 < value

    :param str alias: A valid python identifier serving as a settings alias.
        Lower snake case preferred, e.g. ``my_setting``.
    :param str modulename: The module name where the setting is declared,
        typically ``__name__``.
    :param str deepname: A ``.``-separated string of names. E.g. for a module
        constant, use ``MY_CONSTANT``. For a class attributue, use
        ``MyClass.my_attribute``.
    :param callable validator: Optional validator that inputs a value,
        possibly raises validation errors, and returns None.
    """
    global __doc__
    assert isinstance(alias, str)
    assert alias.isidentifier()
    assert isinstance(modulename, str)
    assert isinstance(deepname, str)
    _REGISTRY[alias] = modulename, deepname, validator

    # Add default value to module docstring.
    __doc__ = _doc_template.format(
        defaults="\n".join(f"- {a} = {get(a)}" for a in sorted(_REGISTRY))
    )

    # Support use as a decorator on an optional user-provided validator.
    if validator is None:
        # Return a decorator, but its fine if user discards this.
        return functools.partial(register, alias, modulename, deepname)
    else:
        # Test current value passes validation.
        validator(get(alias))
        return validator
