# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal demo: read-only KV-cache eviction explanation.

Uses a small-capacity fake block pool (6 blocks, 5 usable) to simulate:
  1. A cache hit (must stay silent -- no eviction events).
  2. An expansion failure (no eviction on a failed allocation).
  3. Request cancellation.
  4. LRU eviction under pool pressure (pool_reuse).
  5. Explicit eviction and prefix-cache reset.

Run:
    python examples/kv_cache_eviction_demo.py
"""

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

BLOCK_SIZE = 4
NUM_BLOCKS = 6  # 5 usable + the null block.


def make_manager():
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


def make_request(request_id, token_ids):
    return Request(
        request_id=request_id,
        prompt_token_ids=token_ids,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=17),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def main():
    init_none_hash(sha256)
    manager = make_manager()
    explainer = manager.eviction_explainer

    print("=" * 70)
    print("KV-cache eviction explanation demo")
    print(f"Pool: {NUM_BLOCKS} blocks ({NUM_BLOCKS - 1} usable)")
    print("=" * 70)

    # 1. Cache miss then hit (hit path must be silent).
    req_a = make_request("req_a", [10] * 8)
    computed, num_computed, _ = manager.get_computed_blocks(req_a)
    manager.allocate_slots(req_a, 8, 0, computed)
    req_b = make_request("req_b", [10] * 12)
    computed, num_computed, _ = manager.get_computed_blocks(req_b)
    manager.allocate_slots(req_b, 4, 8, computed)
    print("\n[1] After cache miss + hit:")
    print(f"    computed tokens for req_b: {num_computed} (hit, silent)")
    print(f"    eviction records: {len(explainer.history)} (expect 0)")

    # 2. Expansion failure (not enough free blocks).
    req_c = make_request("req_c", [20] * 20)
    computed, num_computed, _ = manager.get_computed_blocks(req_c)
    result = manager.allocate_slots(req_c, 20, 0, computed)
    print("\n[2] After expansion failure:")
    print(f"    allocate result: {result} (None = failed)")
    print(f"    eviction records: {len(explainer.history)} (expect 0)")

    # 3. Cancellation frees req_a and req_b.
    manager.free(req_a)
    manager.free(req_b)
    print("\n[3] After cancelling req_a + req_b:")
    print(f"    free blocks: {manager.block_pool.get_num_free_blocks()} "
          f"(expect {NUM_BLOCKS - 1})")

    # 4. LRU eviction under pool pressure.
    manager.new_step_starts()
    req_d = make_request("req_d", [30] * 16)
    computed, num_computed, _ = manager.get_computed_blocks(req_d)
    manager.allocate_slots(req_d, 16, 0, computed)
    print("\n[4] After req_d allocation (pool pressure):")
    print(f"    eviction records: {len(explainer.history)} (expect 1)")
    record = explainer.history[0]
    print(f"    reason: {record.reason.value}, step: {record.step}")
    for blk in record.blocks:
        print(f"    - block {blk.block_id}: last_ref={blk.last_request_id}"
              f"@{blk.last_access_step}, ref_cnt={blk.ref_cnt}")

    # 5. Explicit eviction + reset.
    manager.block_pool.evict_blocks({1})
    manager.free(req_d)
    manager.new_step_starts()
    req_e = make_request("req_e", [30] * 20)  # hit on req_d's prefix
    computed, num_computed, _ = manager.get_computed_blocks(req_e)
    manager.allocate_slots(req_e, 4, 16, computed)
    manager.free(req_e)
    manager.new_step_starts()
    manager.reset_prefix_cache()
    print("\n[5] After explicit evict + hit + reset:")
    print(f"    eviction records: {len(explainer.history)} (expect 3)")
    for record in explainer.history:
        print(f"    step {record.step} [{record.reason.value}]: "
              f"{[b.block_id for b in record.blocks]}")

    print("\n" + "=" * 70)
    print("Human-readable explanation:")
    print("=" * 70)
    print(explainer.explain())

    print("\n" + "=" * 70)
    print("Boundary checks:")
    print("=" * 70)
    # The explainer holds no live block objects (only primitives).
    for record in explainer.history:
        for blk in record.blocks:
            assert isinstance(blk.block_id, int)
            assert isinstance(blk.ref_cnt, int)
            assert not hasattr(blk, "prev_free_block")
    print("  [OK] Explainer holds no live block objects (only primitives).")
    # The hit path never recorded an eviction.
    assert explainer.history[0].reason.value == "pool_reuse"
    print("  [OK] First eviction is pool_reuse (hit path stayed silent).")


if __name__ == "__main__":
    main()
