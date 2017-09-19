import common


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "init(rng_seed): initialize the RNG using the seed provided")


def pytest_runtest_setup(item):
    test_initialize_marker = item.get_marker("init")
    if test_initialize_marker:
        rng_seed = test_initialize_marker.kwargs["rng_seed"]
        common.set_rng_seed(rng_seed)
