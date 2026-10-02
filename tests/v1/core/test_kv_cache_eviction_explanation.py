# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only KV-cache eviction explanations.

These tests use a tiny in-process block pool (6 physical blocks, block size 4)
to exercise, deterministically and without any GPU/container dependency:

  1. a prefix-cache *hit* -- must NOT produce an eviction event;
  2. a capacity *allocation failure* (``allocate_slots`` -> ``None``) -- must
     neither allocate nor evict;
  3. an *eviction* driven by a new allocation -- an ``EvictionEvent`` lists the
     evicted blocks and their last referencing request;
  4. request *cancellation* (``manager.free``) -- no event on release, and the
     released cached blocks become the next eviction candidates;
  5. explicit ``evict_blocks`` -- an ``explicit_eviction`` event;
  6. the safety boundary: an explanation is an immutable primitive snapshot,
     never a live block, and remains valid after the block is reused.

Run directly (prints a small scenario report) or via pytest:

    python tests/v1/core/test_kv_cache_eviction_explanation.py
    python -m pytest tests/v1/core/test_kv_cache_eviction_explanation.py -q
"""

import dataclasses
from collections.abc import Callable

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import (
    EVICTION_REASON_ALLOCATION,
    EVICTION_REASON_EXPLICIT,
    BlockPool,
)
from vllm.v1.core.kv_cache_manager import (
    EvictedBlockInfo,
    EvictionEvent,
    KVCacheManager,
)
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

BLOCK_SIZE = 4
NUM_BLOCKS = 6  # 5 usable + 1 null block.


@pytest.fixture(autouse=True)
def _init_hash_fn():
    init_none_hash(sha256)


def make_manager(num_blocks: int = NUM_BLOCKS) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=1024,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
    )


def make_request(
    request_id: str,
    tokens: list[int],
    hash_fn: Callable = sha256,
) -> Request:
    sampling_params = SamplingParams(max_tokens=16)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, hash_fn),
    )


def allocate_and_cache(manager: KVCacheManager, request: Request):
    """Admit a fresh prompt and cache all of its full blocks."""
    request.status = RequestStatus.RUNNING
    blocks = manager.allocate_slots(request, num_new_tokens=request.num_tokens)
    assert blocks is not None, "test setup allocation must succeed"
    return blocks


def test_hit_allocation_failure_eviction_and_cancellation():
    """The main acceptance scenario, run against the real KVCacheManager."""
    manager = make_manager()
    # 5 usable blocks; keep the explanation queue empty at the start.
    assert manager.take_eviction_events() == []

    # ---- Phase 1: request A fills 3 blocks (2 full + 1 partial) and the
    # full blocks enter the prefix cache on free. No eviction is needed. ----
    a = make_request("req-a", list(range(10)))
    allocate_and_cache(manager, a)
    manager.free(a)
    assert manager.take_eviction_events() == []

    # ---- Phase 2: request B shares A's first two full blocks. The hit is
    # adopted (touch pins the cached blocks) and B grows into the rest. ----
    b = make_request("req-b", list(range(10)) + [100, 101])
    b.status = RequestStatus.RUNNING
    # max_cache_hit_length = prompt_length - 1: a 9-token-equivalent prefix
    # still reports the two full cached blocks while recomputing its tail.
    hit_blocks, num_computed, _ = manager.get_computed_blocks(
        make_request("req-b-lookup", list(range(10)) + [100, 101])
    )
    assert num_computed == 2 * BLOCK_SIZE
    assert len(hit_blocks.blocks[0]) == 2
    # The lookup path is read-only: it must never evict anything.
    assert manager.take_eviction_events() == []
    assert (
        manager.allocate_slots(
            b,
            num_new_tokens=12 - num_computed,
            num_new_computed_tokens=num_computed,
            new_computed_blocks=hit_blocks,
        )
        is not None
    )

    # ---- Phase 3: capacity expansion failure. A 5-block request with no
    # prefix hit cannot fit (only 2 blocks are free). Nothing is allocated
    # and nothing is evicted: the failed attempt is side-effect free. ----
    big = make_request("req-too-big", list(range(20)))
    big.status = RequestStatus.RUNNING
    assert manager.allocate_slots(big, num_new_tokens=20) is None
    assert manager.take_eviction_events() == []
    # B is untouched: it still holds 3 blocks and references the two
    # prefix-cached blocks (ref_cnt >= 1 keeps them out of the free queue).
    assert manager.block_pool.get_num_free_blocks() == 2
    assert all(blk.ref_cnt >= 1 for blk in manager.get_blocks("req-b").blocks[0][:2])

    # ---- Phase 4: cancel B. Releasing references never evicts, so no event;
    # its pinned blocks rejoin the cached free queue as eviction candidates.
    manager.free(b)
    assert manager.take_eviction_events() == []
    assert manager.block_pool.get_num_free_blocks() == 5

    # ---- Phase 5: request C (different prefix) needs 4 fresh blocks. The
    # ---- Phase 5: request C (different prefix) needs 4 fresh blocks. All
    # free blocks now carry hashes (B filled the partial tail), so the two
    # blocks at the FIFO tail of the cached queue are evicted. ----
    c = make_request("req-c", list(range(1000, 1016)))
    allocate_and_cache(manager, c)
    events = manager.take_eviction_events()
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, EvictionEvent)
    assert event.reason == EVICTION_REASON_ALLOCATION
    assert event.request_id == "req-c"
    # Two oldest cached blocks are evicted, FIFO across the cached tail.
    assert event.block_ids == (3, 2)
    for info in event.blocks:
        assert isinstance(info, EvictedBlockInfo)
        # Both blocks were last referenced by req-b (the hit pinned them and
        # cancellation released them).
        assert info.last_request_id == "req-b"
        assert info.group_id == 0
        assert info.num_hash_tokens and info.num_hash_tokens >= BLOCK_SIZE
        assert isinstance(info.block_hash, bytes)
    rendered = event.format()
    assert "req-c" in rendered and "req-b" in rendered

    # ---- Phase 6: cancel C; release again produces no event. ----
    manager.free(c)
    assert manager.block_pool.get_num_free_blocks() == 5
    assert manager.take_eviction_events() == []


def test_explicit_eviction_explains_last_referrer():
    """``evict_blocks`` emits one explicit_eviction event per call."""
    manager = make_manager()
    a = make_request("req-a", list(range(10)))
    allocate_and_cache(manager, a)
    cached_ids = [
        blk.block_id
        for blk in manager.get_blocks("req-a").blocks[0]
        if blk.block_hash is not None
    ]
    assert len(cached_ids) == 2
    # Blocks still referenced by req-a only lose their cache hash; the blocks
    # themselves stay allocated.
    manager.evict_blocks(set(cached_ids))
    events = manager.take_eviction_events()
    assert len(events) == 1
    assert events[0].reason == EVICTION_REASON_EXPLICIT
    assert events[0].request_id is None
    assert set(events[0].block_ids) == set(cached_ids)
    assert {b.last_request_id for b in events[0].blocks} == {"req-a"}
    # Draining is destructive.
    assert manager.take_eviction_events() == []
    manager.free(a)


def test_explanation_is_a_snapshot_not_a_live_block():
    """An explanation must not hold a released block: after reuse it still
    reports the block identity captured at eviction time (frozen primitives)."""
    manager = make_manager()
    a = make_request("req-a", list(range(10)))
    allocate_and_cache(manager, a)
    cached_physical = [
        blk.block_id
        for blk in manager.get_blocks("req-a").blocks[0]
        if blk.block_hash is not None
    ]
    manager.free(a)

    c = make_request("req-c", list(range(2000, 2016)))  # 4 fresh blocks
    allocate_and_cache(manager, c)
    events = manager.take_eviction_events()
    assert len(events) == 1
    frozen = events[0].blocks[0]
    # Frozen, primitive, hashable.
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        frozen.block_id = -1  # type: ignore[misc]
    assert isinstance(hash(frozen), int)
    assert frozen.block_id in cached_physical
    assert frozen.last_request_id == "req-a"

    # That physical block now belongs to req-c with new contents, while the
    # snapshot keeps describing the earlier eviction.
    live = manager.block_pool.blocks[frozen.block_id]
    assert live.ref_cnt == 1
    assert live.last_request_id == "req-c"
    manager.free(c)


def test_reset_prefix_cache_is_not_an_eviction():
    """A bulk reset invalidates everything but must not masquerade as an
    eviction explanation."""
    manager = make_manager()
    a = make_request("req-a", list(range(8)))
    allocate_and_cache(manager, a)
    manager.free(a)
    assert manager.reset_prefix_cache() is True
    assert manager.take_eviction_events() == []


def test_block_pool_level_last_referrer_tracking():
    """Minimal direct check against the small fake pool: a cache hit (touch)
    records no eviction event, and allocation eviction names the last
    referrer and the requesting request."""
    pool = BlockPool(
        num_gpu_blocks=NUM_BLOCKS,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
    )
    assert pool.take_eviction_events() == []

    key0 = b"hash-a" + (0).to_bytes(4, "big")
    key1 = b"hash-b" + (0).to_bytes(4, "big")
    first = pool.get_new_blocks(2, request_id="req-1")
    for blk, key in zip(first, (key0, key1), strict=True):
        blk.set_block_hash(key, num_tokens=BLOCK_SIZE)
        pool.cached_block_hash_to_block.insert(key, blk)
    pool.free_blocks(reversed(first), request_id="req-1")
    assert pool.take_eviction_events() == []

    # req-2 hits the first cached block: touch pins it, records no eviction.
    hit = pool.cached_block_hash_to_block.get_one_block(key0)
    assert hit is first[0]
    pool.touch([hit])
    assert pool.take_eviction_events() == []

    # req-3 demands 4 blocks: the free queue yields 4; the still-cached,
    # unpinned block (key1) is among them and gets evicted. The touched
    # block (key0) is pinned and cannot be selected.
    fresh = pool.get_new_blocks(4, request_id="req-3")
    events = pool.take_eviction_events()
    assert [e.reason for e in events] == [EVICTION_REASON_ALLOCATION]
    evicted = events[0].blocks
    assert [b.block_id for b in evicted] == [first[1].block_id]
    assert evicted[0].last_request_id == "req-1"
    assert events[0].request_id == "req-3"
    assert len(fresh) == 4
    assert all(b.last_request_id == "req-3" for b in fresh)
    pool.free_blocks(reversed(fresh), request_id="req-3")


def _print_demo() -> None:
    """Print the acceptance scenario for ``python <this_file>``."""
    init_none_hash(sha256)
    manager = make_manager()
    print(
        f"pool: {NUM_BLOCKS} blocks ({NUM_BLOCKS - 1} usable), "
        f"block_size={BLOCK_SIZE}, prefix caching on"
    )

    a = make_request("req-a", list(range(10)))
    allocate_and_cache(manager, a)
    manager.free(a)
    print("after req-a free: eviction events:", manager.take_eviction_events())

    b = make_request("req-b", list(range(10)) + [1000, 101])
    b.status = RequestStatus.RUNNING
    hit_blocks, hits, _ = manager.get_computed_blocks(b)
    print(
        f"prefix hit: {hits // BLOCK_SIZE} cached block(s); "
        f"eviction events: {manager.take_eviction_events()}"
    )
    assert (
        manager.allocate_slots(
            b,
            num_new_tokens=12 - hits,
            num_new_computed_tokens=hits,
            new_computed_blocks=hit_blocks,
        )
        is not None
    )

    big = make_request("req-too-big", list(range(20)))
    big.status = RequestStatus.RUNNING
    result = manager.allocate_slots(big, num_new_tokens=20)
    print(
        f"allocation failure (result={result}); "
        f"eviction events: {manager.take_eviction_events()}"
    )

    manager.free(b)
    print("req-b cancelled; eviction events:", manager.take_eviction_events())

    c = make_request("req-c", list(range(1000, 1016)))
    allocate_and_cache(manager, c)
    for event in manager.take_eviction_events():
        print("EVICTION:", event.format())
        for info in event.blocks:
            print(
                f"  - block_id={info.block_id} group={info.group_id} "
                f"tokens={info.num_hash_tokens} last_request={info.last_request_id}"
            )
    manager.free(c)


if __name__ == "__main__":
    _print_demo()
