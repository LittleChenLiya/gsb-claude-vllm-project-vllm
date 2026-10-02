# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the scheduler's read-only queuing-reason snapshot.

These tests drive the real ``KVCacheManager`` on CPU with a small, controlled
``num_blocks`` (a CPU-level fake KV setup) to create three kinds of blocking --
token budget, KV capacity, and priority -- plus cancellation, and verify the
scheduling order and the snapshot contents.
"""

import pytest

from vllm.v1.core.sched.scheduler import QueuingReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .test_scheduler import create_requests_with_priority
from .utils import (
    create_requests,
    create_scheduler,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _patch_cpu_max_threads(monkeypatch):
    """Work around ``get_max_threads`` not supporting Windows.

    ``CpuPlatform.check_and_update_config`` calls ``get_max_threads`` to set
    ``NUMEXPR_MAX_THREADS``; on Windows ``os.sched_getaffinity`` is absent, so
    it raises ``NotImplementedError``. Patch it to a fixed value for the tests.
    """
    import vllm.platforms.cpu as cpu_platform

    monkeypatch.setattr(cpu_platform, "get_max_threads", lambda *a, **k: 8)


def _make_request(num_tokens: int, req_id: str, block_size: int = 16):
    return create_requests(
        num_requests=1,
        num_tokens=num_tokens,
        block_size=block_size,
        req_ids=[req_id],
    )[0]


def test_token_budget_blocking():
    """A waiting request is blocked because the per-step token budget is exhausted."""
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        num_blocks=100,  # plenty of KV blocks so KV capacity is not the constraint
        block_size=16,
    )
    # req_a's prefill is larger than the token budget, so it consumes the whole
    # budget in one chunk; req_b then cannot be admitted this step.
    req_a = _make_request(200, "a")
    req_b = _make_request(16, "b")

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    output = scheduler.schedule()

    # Only req_a is scheduled (chunked to the token budget).
    assert set(output.num_scheduled_tokens) == {req_a.request_id}
    assert output.num_scheduled_tokens[req_a.request_id] == 100
    # req_b remains queued, blocked by the token budget.
    assert req_b.status == RequestStatus.WAITING
    snapshot = scheduler.get_queuing_reason_snapshot()
    assert snapshot == {req_b.request_id: QueuingReason.TOKEN_BUDGET}


def test_kv_capacity_blocking():
    """A waiting request is blocked because the KV cache has no free blocks."""
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,  # plenty of token budget so tokens are not the constraint
        num_blocks=3,
        block_size=16,
    )
    req_a = _make_request(32, "a")
    req_b = _make_request(32, "b")

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    output = scheduler.schedule()

    # req_a is admitted (2 blocks); req_b cannot allocate (needs 2, only 1 free).
    assert set(output.num_scheduled_tokens) == {req_a.request_id}
    assert req_b.status == RequestStatus.WAITING
    snapshot = scheduler.get_queuing_reason_snapshot()
    assert snapshot == {req_b.request_id: QueuingReason.KV_CAPACITY}


def test_priority_blocking():
    """A low-priority running request is preempted and queued due to priority."""
    # num_blocks=5 -> 4 usable (block 0 is the reserved null block). Both
    # 32-token requests fit in the first schedule (2 blocks each), but the
    # second schedule's decode needs a 5th block, forcing a preemption.
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        num_blocks=5,
        block_size=16,
        scheduling_policy="priority",
    )
    # req_b has higher priority (lower value) than req_c.
    req_b, req_c = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 10],
        num_tokens=32,
        block_size=16,
        req_ids=["b", "c"],
    )

    scheduler.add_request(req_b)
    scheduler.add_request(req_c)

    # First schedule: both admitted in priority order (b then c).
    output1 = scheduler.schedule()
    assert list(output1.num_scheduled_tokens) == ["b", "c"]
    assert req_b.status == RequestStatus.RUNNING
    assert req_c.status == RequestStatus.RUNNING
    # Nothing queued yet.
    assert scheduler.get_queuing_reason_snapshot() == {}

    # Simulate a decode step: each request generates one output token. This
    # grows num_tokens so the next schedule must decode.
    model_runner_output = ModelRunnerOutput(
        req_ids=["b", "c"],
        req_id_to_index={"b": 0, "c": 1},
        sampled_token_ids=[[0], [0]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output1, model_runner_output)

    # Second schedule: b's decode needs a 3rd block. The KV cache is full, so the
    # lowest-priority running request (c) is preempted. Because a preemption
    # occurred, the waiting-admission pass is skipped this step, so c keeps the
    # PRIORITY reason from its preemption.
    output2 = scheduler.schedule()
    assert list(output2.num_scheduled_tokens) == ["b"]
    assert req_c.status == RequestStatus.PREEMPTED
    snapshot = scheduler.get_queuing_reason_snapshot()
    assert snapshot == {req_c.request_id: QueuingReason.PRIORITY}


def test_cancellation_removes_from_snapshot():
    """A cancelled request must not remain in the queuing snapshot."""
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        num_blocks=100,
        block_size=16,
    )
    req_a = _make_request(200, "a")
    req_b = _make_request(16, "b")

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    scheduler.schedule()

    # req_b is queued with TOKEN_BUDGET.
    snapshot = scheduler.get_queuing_reason_snapshot()
    assert req_b.request_id in snapshot

    # Cancel req_b (e.g. client disconnect).
    scheduler.finish_requests(req_b.request_id, RequestStatus.FINISHED_ABORTED)

    # req_b must be gone from the snapshot.
    snapshot = scheduler.get_queuing_reason_snapshot()
    assert req_b.request_id not in snapshot
    assert snapshot == {}


def test_snapshot_does_not_change_queue_order():
    """Reading the snapshot must not mutate the queuing order."""
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        num_blocks=100,
        block_size=16,
    )
    req_a = _make_request(200, "a")
    req_b = _make_request(16, "b")
    req_c = _make_request(16, "c")

    for req in (req_a, req_b, req_c):
        scheduler.add_request(req)

    scheduler.schedule()

    before = [r.request_id for r in scheduler.waiting]
    # Read the snapshot multiple times; it must be a pure read.
    _ = scheduler.get_queuing_reason_snapshot()
    _ = scheduler.get_queuing_reason_snapshot()
    after = [r.request_id for r in scheduler.waiting]
    assert before == after


def test_snapshot_is_read_only_copy():
    """Mutating the returned snapshot must not affect the scheduler."""
    scheduler = create_scheduler(
        max_num_batched_tokens=100,
        num_blocks=100,
        block_size=16,
    )
    req_a = _make_request(200, "a")
    req_b = _make_request(16, "b")

    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    scheduler.schedule()

    snapshot = scheduler.get_queuing_reason_snapshot()
    assert snapshot == {req_b.request_id: QueuingReason.TOKEN_BUDGET}

    # Mutate the returned copy; the scheduler's internal state is unaffected.
    snapshot.clear()
    snapshot[req_a.request_id] = QueuingReason.PRIORITY
    assert scheduler.get_queuing_reason_snapshot() == {
        req_b.request_id: QueuingReason.TOKEN_BUDGET
    }
