# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the scheduler's read-only queued-reason snapshot.

The snapshot records, for every request that remains in the waiting queue
after a scheduling step, one of three reasons:

* ``QueuedReason.TOKEN_BUDGET`` - per-step token budget exhausted
* ``QueuedReason.KV_CAPACITY`` - the (fake, CPU-level) KV cache manager could
  not allocate blocks
* ``QueuedReason.PRIORITY`` - waiting for an active request slot / queue order

These tests use the normal CPU test scheduler plus a tiny fake KV-cache-manager
wrapper that can script ``allocate_slots`` failures, so the three blocking
conditions and request cancellation are fully reproducible on CPU.
"""

import dataclasses

import pytest

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import (
    QueuedReason,
    QueuedRequestSnapshot,
)
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


class FakeKVCacheManager:
    """CPU-level controllable KV cache manager.

    Delegates everything to the real ``KVCacheManager`` except
    ``allocate_slots``, whose result can be scripted: while ``deny`` is True
    every allocation returns None (simulating exhausted KV capacity).
    """

    def __init__(self, real_manager: KVCacheManager):
        object.__setattr__(self, "_real", real_manager)
        object.__setattr__(self, "deny", False)
        object.__setattr__(self, "allocations", 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        if name in ("_real", "deny", "allocations"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)

    @property
    def usage(self):
        return self._real.usage

    def allocate_slots(self, *args, **kwargs):
        object.__setattr__(self, "allocations", self.allocations + 1)
        if object.__getattribute__(self, "deny"):
            return None
        return self._real.allocate_slots(*args, **kwargs)


def _make_scheduler(**kwargs):
    scheduler = create_scheduler(**kwargs)
    fake = FakeKVCacheManager(scheduler.kv_cache_manager)
    scheduler.kv_cache_manager = fake
    return scheduler, fake


def _new_requests(num, num_tokens, req_ids, priorities=None, arrivals=None):
    requests = create_requests(
        num,
        num_tokens=num_tokens,
        req_ids=req_ids,
        block_size=16,
    )
    for i, req in enumerate(requests):
        if priorities is not None:
            req.priority = priorities[i]
        if arrivals is not None:
            req.arrival_time = arrivals[i]
    return requests


def _new_req_ids(scheduler_output):
    return [d.req_id for d in scheduler_output.scheduled_new_reqs]


def _ids(snapshot):
    return [s.request_id for s in snapshot]


def _reason_map(snapshot):
    return {s.request_id: s.reason for s in snapshot}


# ---------------------------------------------------------------------------
# 1. TOKEN_BUDGET blocking
# ---------------------------------------------------------------------------
def test_snapshot_token_budget_blocking_preserves_fcfs_order():
    scheduler, _ = _make_scheduler(
        max_num_seqs=16,
        max_num_active_seqs=16,
        max_num_batched_tokens=20,
        enable_chunked_prefill=True,
        num_blocks=10000,
        scheduling_policy="fcfs",
    )
    # Each request needs 20 prompt tokens = the whole step token budget, so
    # exactly one is admitted per step.
    reqs = _new_requests(3, num_tokens=20, req_ids=["a", "b", "c"])
    for req in reqs:
        scheduler.add_request(req)

    out = scheduler.schedule()
    assert _new_req_ids(out) == ["a"]

    snapshot = scheduler.get_queued_reason_snapshot()
    # Snapshot is a read-only tuple of frozen entries in FCFS queue order.
    assert isinstance(snapshot, tuple)
    assert dataclasses.is_dataclass(snapshot[0])
    assert _ids(snapshot) == ["b", "c"]
    reasons = _reason_map(snapshot)
    assert reasons["b"] == QueuedReason.TOKEN_BUDGET
    assert reasons["c"] == QueuedReason.PRIORITY

    # Taking the snapshot did not change the queue: it repeats identically.
    assert _ids(scheduler.get_queued_reason_snapshot()) == ["b", "c"]

    # Scheduling does not reorder: one admission per step in FCFS order.
    out = scheduler.schedule()
    assert _new_req_ids(out) == ["b"]
    out = scheduler.schedule()
    assert _new_req_ids(out) == ["c"]


# ---------------------------------------------------------------------------
# 2. KV_CAPACITY blocking (scripted fake KV manager)
# ---------------------------------------------------------------------------
def test_snapshot_kv_capacity_blocking():
    # Only 4 blocks of 16 tokens; a 20-token prompt needs 2 blocks.
    scheduler, fake = _make_scheduler(
        max_num_seqs=16,
        max_num_active_seqs=16,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        num_blocks=4,
        scheduling_policy="fcfs",
    )
    reqs = _new_requests(2, num_tokens=20, req_ids=["x", "y"])
    for req in reqs:
        scheduler.add_request(req)

    # First step: admit x (consumes 2 blocks).
    out = scheduler.schedule()
    assert _new_req_ids(out) == ["x"]
    assert scheduler.get_num_unfinished_requests() == 2

    # Deny all allocations now: y cannot get KV blocks.
    fake.deny = True
    out = scheduler.schedule()
    assert _new_req_ids(out) == []
    snapshot = scheduler.get_queued_reason_snapshot()
    assert _ids(snapshot) == ["y"]
    assert snapshot[0].reason == QueuedReason.KV_CAPACITY

    # y stayed queued (snapshot is non-mutating), status untouched.
    assert [r.request_id for r in scheduler.waiting] == ["y"]
    assert scheduler.requests["y"].status == RequestStatus.WAITING

    # Once capacity returns and x is finished, y is admitted in its position.
    fake.deny = False
    scheduler.finish_requests(["x"], RequestStatus.FINISHED_ABORTED)
    out = scheduler.schedule()
    assert _new_req_ids(out) == ["y"]
    assert scheduler.get_queued_reason_snapshot() == ()


# ---------------------------------------------------------------------------
# 3. PRIORITY blocking (active-slot cap) + priority ordering
# ---------------------------------------------------------------------------
def test_snapshot_priority_blocking_and_order():
    scheduler, _ = _make_scheduler(
        max_num_seqs=1,
        max_num_active_seqs=1,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        num_blocks=10000,
        scheduling_policy="priority",
    )
    # low priority arrives first; high priority arrives second.
    reqs = _new_requests(
        2,
        num_tokens=10,
        req_ids=["low", "high"],
        priorities=[10, 0],
        arrivals=[1.0, 2.0],
    )
    for req in reqs:
        scheduler.add_request(req)

    out = scheduler.schedule()
    # One active slot: the higher-priority "high" is admitted first.
    assert _new_req_ids(out) == ["high"]

    snapshot = scheduler.get_queued_reason_snapshot()
    assert _ids(snapshot) == ["low"]
    assert snapshot[0].reason == QueuedReason.PRIORITY
    assert snapshot[0].priority == 10

    # Release the slot: low is admitted next, keeping its queue position.
    scheduler.finish_requests(["high"], RequestStatus.FINISHED_ABORTED)
    out = scheduler.schedule()
    assert _new_req_ids(out) == ["low"]


# ---------------------------------------------------------------------------
# 4. Cancellation never leaves the request in the snapshot
# ---------------------------------------------------------------------------
def test_snapshot_excludes_cancelled_requests():
    scheduler, _ = _make_scheduler(
        max_num_seqs=16,
        max_num_active_seqs=16,
        max_num_batched_tokens=20,
        enable_chunked_prefill=True,
        num_blocks=10000,
        scheduling_policy="fcfs",
    )
    reqs = _new_requests(3, num_tokens=20, req_ids=["r1", "r2", "r3"])
    for req in reqs:
        scheduler.add_request(req)

    scheduler.schedule()
    assert _ids(scheduler.get_queued_reason_snapshot()) == ["r2", "r3"]

    # Abort a queued request before the next step: the prior snapshot is
    # pruned synchronously.
    scheduler.finish_requests(["r2"], RequestStatus.FINISHED_ABORTED)
    snapshot = scheduler.get_queued_reason_snapshot()
    assert _ids(snapshot) == ["r3"]
    assert "r2" not in scheduler.requests

    # Next step admits r3 (r2 never sneaks back in), leaving no queued reqs.
    out = scheduler.schedule()
    assert _new_req_ids(out) == ["r3"]
    assert scheduler.get_queued_reason_snapshot() == ()

    # Cancelling all remaining requests empties snapshot and queues.
    scheduler.finish_requests(None, RequestStatus.FINISHED_ABORTED)
    assert scheduler.get_queued_reason_snapshot() == ()
    assert not scheduler.has_unfinished_requests()


def test_snapshot_entries_are_immutable_copies():
    scheduler, _ = _make_scheduler(
        max_num_batched_tokens=20,
        scheduling_policy="fcfs",
    )
    reqs = _new_requests(2, num_tokens=20, req_ids=["p", "q"])
    for req in reqs:
        scheduler.add_request(req)
    scheduler.schedule()

    snap = scheduler.get_queued_reason_snapshot()
    assert len(snap) == 1
    assert isinstance(snap[0], QueuedRequestSnapshot)
    for field in dataclasses.fields(QueuedRequestSnapshot):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(snap[0], field.name, None)
