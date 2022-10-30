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
        "pyro.distributions.torch",  # module
        "Binomial.approx_sample_thresh",  # deep name
    )

    # Register a new setting on a user-provided validator.
    @pyro.settings.register(
        "binomial_approx_sample_thresh",  # alias
        "pyro.distributions.torch",  # module
        "Binomial.approx_sample_thresh",  # deep name
    )
    def validate_thresh(thresh):  # called each time setting is set
        assert isinstance(thresh, float)
        assert thresh > 0
"""

# This library must have no other dependencies on pyro.
import functools
from contextlib import contextmanager
from importlib import import_module
from typing import Any, Callable, Dict, Optional, Tuple

# Global registry mapping alias:str to (modulename, deepname, validator)
# triples where deepname may have dots to indicate e.g. class variables.
_REGISTRY: Dict[str, Tuple[str, str, Optional[Callable]]] = {}


def register(
    alias: str,
    modulename: str,
    deepname: str,
    validator: Optional[Callable] = None,
) -> Callable:
    assert isinstance(alias, str)
    assert isinstance(modulename, str)
    assert isinstance(deepname, str)
    _REGISTRY[alias] = modulename, deepname, validator

    # Support use as a decorator on an optional user-provided validator.
    if validator is None:
        # Smoke test to check that setting exists.
        get(alias)
        # Return a decorator, but its fine if user discards this.
        return functools.partial(register, alias, modulename, deepname)
    else:
        # Test current value passes validation.
        validator(get(alias))
        return validator


def get(alias: Optional[str] = None) -> Any:
    """
    Gets one or all global settings.
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
    """
    Sets one or more settings.
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
def context(**kwargs):
    """
    Context manager to temporarily override one or more settings.
    """
    old = {alias: get(alias) for alias in kwargs}
    try:
        set(**kwargs)
        yield
    finally:
        set(**old)
