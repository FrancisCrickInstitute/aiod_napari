import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow/compute-intensive tests (excluded by default).",
    )
    parser.addoption(
        "--full-models",
        action="store_true",
        default=False,
        help="Run the full model suite (all tasks, models, and variants) instead of the smoke test.",
    )
    parser.addoption(
        "--one-per-model",
        action="store_true",
        default=False,
        help="Run one variant per model (first available) for every task, instead of the smoke test.",
    )
    parser.addoption(
        "--one-model",
        action="store",
        nargs="?",
        const="mito,Empanada,MitoNet v1",
        default=None,
        metavar="TASK,MODEL,VARIANT",
        help='Run a single specific combination, e.g. pytest --one-model "mito,Empanada,MitoNet v1". Bare --one-model uses the smoke-test default.',
    )
    parser.addoption(
        "--pipeline-timeout",
        action="store",
        default=180,
        type=int,
        metavar="SECONDS",
        help="Seconds to wait for the inference pipeline before the test is failed (default: 300).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test; use --run-slow to run")
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(skip_slow)
