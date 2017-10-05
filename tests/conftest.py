import pyro


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "init(rng_seed): initialize the RNG using the seed provided.")
    config.addinivalue_line("markers",
                            "stage(NAME): mark test to run when testing stage matches NAME.")


def pytest_runtest_setup(item):
    test_initialize_marker = item.get_marker("init")
    if test_initialize_marker:
        rng_seed = test_initialize_marker.kwargs["rng_seed"]
        pyro.set_rng_seed(rng_seed)


def pytest_addoption(parser):
    parser.addoption("--stage",
                     action="append",
                     metavar="NAME",
                     default=[],
                     help="Only run tests matching the stage NAME.")


def pytest_collection_modifyitems(session, config, items):
    test_stages = set(config.getoption("--stage"))
    if not test_stages:
        test_stages = ["unit"]
    selected_items = []
    deselected_items = []
    for item in items:
        stage_marker = item.get_marker("stage")
        if not stage_marker or stage_marker.args[0] in test_stages:
            selected_items.append(item)
        else:
            deselected_items.append(item)
    config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items
