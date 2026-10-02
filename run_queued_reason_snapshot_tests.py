"""Reproducible local runner for the queued-reason snapshot tests.

Usage:
    python run_queued_reason_snapshot_tests.py [-k EXPRESSION]

Runs the CPU-only tests in
``tests/v1/core/test_queued_reason_snapshot.py`` without any GPU or Docker.

It only (a) forces the CPU platform and (b) on platforms where vLLM's CPU
``get_max_threads`` helper is not implemented (e.g. Windows), falls back to
``os.cpu_count()``. On Linux it is effectively a no-op and the equivalent
plain pytest command is:

    VLLM_TARGET_DEVICE=cpu python -m pytest \
        tests/v1/core/test_queued_reason_snapshot.py -q
"""

import os
import sys

os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

import vllm.platforms.cpu as cpu_platform  # noqa: E402

_orig_get_max_threads = cpu_platform.get_max_threads


def _portable_get_max_threads(pid: int = 0) -> int:
    try:
        return _orig_get_max_threads(pid)
    except NotImplementedError:
        # Windows has no sched_getaffinity and is not the Darwin branch.
        return os.cpu_count() or 1


cpu_platform.get_max_threads = _portable_get_max_threads

import pytest  # noqa: E402

args = ["tests/v1/core/test_queued_reason_snapshot.py", "-q"]
args.extend(sys.argv[1:])
sys.exit(pytest.main(args))
