from __future__ import absolute_import, division, print_function

import warnings

import pyro


def pytest_configure(config):
    config.addinivalue_line("markers",
                            "init(rng_seed): initialize the RNG using the seed provided.")
    config.addinivalue_line("markers",
                            "stage(NAME): mark test to run when testing stage matches NAME.")
    config.addinivalue_line("markers",
                            "disable_validation: disable all validation on this test.")


def pytest_runtest_setup(item):
    pyro.clear_param_store()
    if item.get_closest_marker("disable_validation"):
        pyro.enable_validation(False)
    else:
        pyro.enable_validation(True)
    test_initialize_marker = item.get_closest_marker("init")
    if test_initialize_marker:
        rng_seed = test_initialize_marker.kwargs["rng_seed"]
        pyro.set_rng_seed(rng_seed)


def pytest_addoption(parser):
    parser.addoption("--stage",
                     action="append",
                     metavar="NAME",
                     default=[],
                     help="Only run tests matching the stage NAME.")


def _get_highest_specificity_marker(stage_marker):
    """
    Get the most specific stage marker corresponding to the test. Specificity
    of test function marker is the highest, followed by test class marker and
    module marker.

    :return: List of most specific stage markers for the test.
    """
    is_test_collected = False
    selected_stages = []
    try:
        for marker in stage_marker:
            selected_stages = list(marker.args)
            is_test_collected = True
            break
    except TypeError:
        selected_stages = list(stage_marker.args)
        is_test_collected = True
    if not is_test_collected:
        raise RuntimeError("stage marker needs at least one stage to be specified.")
    return selected_stages


def pytest_collection_modifyitems(config, items):
    test_stages = set(config.getoption("--stage"))
    if not test_stages or "all" in test_stages:
        return
    selected_items = []
    deselected_items = []
    for item in items:
        stage_marker = item.get_closest_marker("stage")
        if not stage_marker:
            selected_items.append(item)
            warnings.warn("No stage associated with the test {}. Will run on each stage invocation.".format(item.name))
            continue
        item_stage_markers = _get_highest_specificity_marker(stage_marker)
        if test_stages.isdisjoint(item_stage_markers):
            deselected_items.append(item)
        else:
            selected_items.append(item)
    config.hook.pytest_deselected(items=deselected_items)
    items[:] = selected_items
