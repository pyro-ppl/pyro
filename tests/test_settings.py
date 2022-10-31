# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pyro import settings

_TEST_SETTING: float = 0.1

pytestmark = pytest.mark.stage("unit")


def test_settings():
    v0 = settings.get()
    assert isinstance(v0, dict)
    assert all(isinstance(alias, str) for alias in v0)
    assert settings.get("validate_distributions_pyro") is True
    assert settings.get("validate_distributions_torch") is True
    assert settings.get("validate_poutine") is True
    assert settings.get("validate_infer") is True


def test_register():
    with pytest.raises(KeyError):
        settings.get("test_setting")

    @settings.register("test_setting", "tests.test_settings", "_TEST_SETTING")
    def _validate(value):
        assert isinstance(value, float)
        assert 0 < value

    # Test simple get and set.
    assert settings.get("test_setting") == 0.1
    settings.set(test_setting=0.2)
    assert settings.get("test_setting") == 0.2
    with pytest.raises(AssertionError):
        settings.set(test_setting=-0.1)

    # Test context manager.
    with settings.context(test_setting=0.3):
        assert settings.get("test_setting") == 0.3
    assert settings.get("test_setting") == 0.2

    # Test decorator.
    @settings.context(test_setting=0.4)
    def fn():
        assert settings.get("test_setting") == 0.4

    fn()
    assert settings.get("test_setting") == 0.2
