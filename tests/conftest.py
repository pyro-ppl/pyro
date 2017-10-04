import pytest

import pyro


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "init(rng_seed): initialize the RNG using the seed provided")
    config.addinivalue_line("markers",
                            "integration_test: mark as integration test")


def pytest_runtest_setup(item):
    test_initialize_marker = item.get_marker("init")
    if test_initialize_marker:
        rng_seed = test_initialize_marker.kwargs["rng_seed"]
        pyro.set_rng_seed(rng_seed)


def pytest_addoption(parser):
    parser.addoption('--run_integration_tests',
                     action='store_true',
                     default=False,
                     help='Do not skip integration tests')


def pytest_collection_modifyitems(config, items):
    run_integration_tests = False
    if config.getoption("--run_integration_tests"):
        run_integration_tests = True
    if run_integration_tests:
        for item in items:
            if "integration_test" not in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Running only integration tests"))
    else:
        for item in items:
            if "integration_test" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Skipping integration tests. "
                                                        "Use --run_integration_tests option to run."))
