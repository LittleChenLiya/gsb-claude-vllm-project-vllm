# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for read-only KV-cache eviction explanations.

Uses a small-capacity fake block pool (real BlockPool with a handful of
blocks) to deterministically exercise: a cache hit (which must stay silent),
an expansion failure, an LRU eviction under pool pressure, and request
cancellation. Also verifies the boundaries: the explainer never holds live
block objects, and the hit path never records an eviction.
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_eviction import (
    EvictionReason,
    KVCacheEvictionExplainer,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4
NUM_BLOCKS = 6  # 5 usable blocks + the null block.


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


def _make_manager():
    config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
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
        max_model_len=4096,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=True,
    )


def _make_request(request_id, token_ids):
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _history(manager):
    return manager.eviction_history


def test_hit_path_is_silent_and_eviction_reports_last_reference():
    manager = _make_manager()
    explainer = manager.eviction_explainer

    # Phase 1: cache miss, then a cache hit (hit path must be silent).
    req_a = _make_request("req_a", [10] * 8)
    computed, num_computed, _ = manager.get_computed_blocks(req_a)
    assert not computed.blocks[0]
    assert num_computed == 0
    blocks = manager.allocate_slots(req_a, 8, 0, computed)
    assert blocks is not None
    assert blocks.get_block_ids() == ([1, 2],)

    req_b = _make_request("req_b", [10] * 12)
    computed, num_computed, _ = manager.get_computed_blocks(req_b)
    assert num_computed == 8
    blocks = manager.allocate_slots(req_b, 4, 8, computed)
    assert blocks is not None
    assert blocks.get_block_ids() == ([3],)

    # The hit path produced no eviction events.
    assert not _history(manager)
    # The hit blocks record req_b as their last accessor.
    assert explainer.last_reference(1) == ("req_b", 0)
    assert explainer.last_reference(2) == ("req_b", 0)

    # Phase 2: expansion failure (not enough free blocks).
    # Only 2 blocks are free; req_c needs 5.
    req_c = _make_request("req_c", [20] * 20)
    computed, num_computed, _ = manager.get_computed_blocks(req_c)
    assert num_computed == 0
    result = manager.allocate_slots(req_c, 20, 0, computed)
    assert result is None  # expansion failed
    # A failed allocation must not evict anything.
    assert not _history(manager)

    # Phase 3: request cancellation frees req_a and req_b.
    manager.free(req_a)
    manager.free(req_b)
    # The cached blocks are now eviction candidates in the free queue.
    assert manager.block_pool.get_num_free_blocks() == 5

    # Phase 4: LRU eviction under pool pressure.
    manager.new_step_starts()  # step 1
    req_d = _make_request("req_d", [30] * 16)
    computed, num_computed, _ = manager.get_computed_blocks(req_d)
    assert num_computed == 0
    blocks = manager.allocate_slots(req_d, 16, 0, computed)
    assert blocks is not None

    history = _history(manager)
    assert len(history) == 1
    record = history[0]
    assert record.reason is EvictionReason.POOL_REUSE
    assert record.step == 1
    # Blocks 3 and 2 were the LRU cached blocks reused for req_d.
    assert [b.block_id for b in record.blocks] == [3, 2]
    for blk in record.blocks:
        # Each evicted block reports its last reference (req_b) and step.
        assert blk.last_request_id == "req_b"
        assert blk.last_access_step == 0
        assert blk.ref_cnt == 1  # handed to req_d
        # The snapshot is a frozen read-only value, not a live block.
        assert type(blk).__name__ == "EvictedBlock"

    # The explainer never holds live block objects: its state is primitives.
    for blk in record.blocks:
        assert isinstance(blk.block_id, int)
        assert isinstance(blk.ref_cnt, int)
    # The evicted blocks are still usable (ref_cnt tracked, no hash pinning).
    assert manager.block_pool.blocks[3].ref_cnt == 1
    assert manager.block_pool.blocks[2].ref_cnt == 1

    # Phase 5: explicit eviction and reset reasons.
    # Block 1 is still cached; evict it explicitly.
    manager.block_pool.evict_blocks({1})
    history = _history(manager)
    assert len(history) == 2
    assert history[1].reason is EvictionReason.EXPLICIT
    assert [b.block_id for b in history[1].blocks] == [1]

    # Cache a follow-up request that extends req_d's prefix (a hit, so the
    # hit path stays silent) and then reset the prefix cache.
    manager.free(req_d)
    manager.new_step_starts()  # step 2
    req_e = _make_request("req_e", [30] * 20)
    computed, num_computed, _ = manager.get_computed_blocks(req_e)
    assert num_computed == 16  # hit on req_d's prefix; no eviction event
    blocks = manager.allocate_slots(req_e, 4, 16, computed)
    assert blocks is not None
    # The hit path produced no eviction events.
    assert len(_history(manager)) == 2
    manager.free(req_e)  # release so reset can clear the cached blocks
    manager.new_step_starts()  # step 3
    assert manager.reset_prefix_cache() is True
    history = _history(manager)
    assert len(history) == 3
    assert history[2].reason is EvictionReason.RESET
    assert {b.block_id for b in history[2].blocks} == {1, 2, 3, 4, 5}

    # The human-readable explanation mentions the reasons and last reference.
    text = manager.explain_evictions()
    assert "pool_reuse" in text
    assert "explicit" in text
    assert "reset" in text
    assert "last_ref=req_b@0" in text


def test_explainer_holds_no_blocks_and_is_bounded():
    explainer = KVCacheEvictionExplainer(history_size=2)

    class _FakeBlock:
        def __init__(self, block_id):
            self.block_id = block_id
            self.block_hash = None
            self.block_hash_num_tokens = None
            self.ref_cnt = 0

    explainer.record_eviction(
        EvictionReason.POOL_REUSE, [_FakeBlock(1), _FakeBlock(2)]
    )
    explainer.record_eviction(EvictionReason.EXPLICIT, [_FakeBlock(3)])
    explainer.record_eviction(EvictionReason.RESET, [_FakeBlock(4)])

    # History is bounded: only the last 2 records survive.
    history = explainer.history
    assert len(history) == 2
    assert history[0].reason is EvictionReason.EXPLICIT
    assert history[1].reason is EvictionReason.RESET

    # No live block objects are retained anywhere in the explainer.
    for record in history:
        for blk in record.blocks:
            assert not hasattr(blk, "prev_free_block")
            assert not hasattr(blk, "next_free_block")

    # Clearing history does not touch any block pool.
    explainer.clear()
    assert explainer.history == ()
    assert "No KV-cache evictions recorded." in explainer.explain()


def test_cancellation_before_allocation_is_safe():
    manager = _make_manager()
    # A request that is freed before it ever allocated blocks is a no-op.
    req = _make_request("req_aborted", [1] * 8)
    manager.free(req)
    assert not _history(manager)
    assert manager.block_pool.get_num_free_blocks() == NUM_BLOCKS - 1
