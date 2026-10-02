# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only explanation of KV-cache evictions.

The :class:`KVCacheEvictionExplainer` is a passive observer of the block
pool: it records *what* was evicted, *why*, and *who last referenced it*,
without ever influencing allocation or retaining live block objects.

Two invariants are enforced by construction:

1. **No retained blocks.** The explainer only stores frozen snapshots
   (block ids, hashes, token counts, request ids). It never holds a
   ``KVCacheBlock``, so it can never pin a freed block or delay its reuse.
2. **Hit path is silent.** Cache hits (``get_computed_blocks`` /
   ``touch``) never record an eviction. Only the actual eviction paths
   (pool reuse, explicit evict, reset, hash replacement) do.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum

from vllm.v1.core.kv_cache_utils import (
    ExternalBlockHash,
    get_block_hash,
    maybe_convert_block_hash,
)


class EvictionReason(str, Enum):
    """Why a prefix-cache block lost its hash."""

    POOL_REUSE = "pool_reuse"
    """The block pool handed the block to a new allocation (LRU eviction)."""
    EXPLICIT = "explicit"
    """``evict_blocks`` was called (e.g. by a KV connector)."""
    RESET = "reset"
    """``reset_prefix_cache`` cleared every hash."""
    HASH_REPLACED = "hash_replaced"
    """A partial->full promotion replaced the block's hash entries."""


@dataclass(frozen=True)
class EvictedBlock:
    """Read-only snapshot of a block evicted from the prefix cache."""

    block_id: int
    group_id: int | None
    block_hash: ExternalBlockHash | None
    num_tokens: int | None
    last_request_id: str | None
    last_access_step: int
    ref_cnt: int


@dataclass(frozen=True)
class EvictionRecord:
    """A single eviction event: one reason, one step, one or more blocks."""

    reason: EvictionReason
    step: int
    blocks: tuple[EvictedBlock, ...]
    explanation: str


class KVCacheEvictionExplainer:
    """A bounded, read-only history of KV-cache evictions.

    Args:
        history_size: Maximum number of eviction records to retain.
    """

    def __init__(self, history_size: int = 64) -> None:
        assert history_size > 0
        self._history: deque[EvictionRecord] = deque(maxlen=history_size)
        # block_id -> (last request that accessed it, step of that access).
        # Only primitives are stored; never a KVCacheBlock object.
        self._last_reference: dict[int, tuple[str | None, int]] = {}
        self._step = 0

    @property
    def current_step(self) -> int:
        return self._step

    def advance_step(self) -> None:
        self._step += 1

    def note_reference(self, block_id: int, request_id: str) -> None:
        """Record that ``request_id`` accessed (cached/hit/allocated) the block."""
        self._last_reference[block_id] = (request_id, self._step)

    def last_reference(self, block_id: int) -> tuple[str | None, int]:
        """Return ``(request_id, step)`` of the last access, or ``(None, -1)``."""
        return self._last_reference.get(block_id, (None, -1))

    def record_eviction(
        self,
        reason: EvictionReason,
        blocks: list,
        *,
        group_id: int | None = None,
        explanation: str | None = None,
    ) -> EvictionRecord:
        """Record an eviction of ``blocks``.

        Only frozen snapshots are retained; the block objects themselves are
        not held, so this can never pin a freed block.
        """
        snapshots: list[EvictedBlock] = []
        for block in blocks:
            last_req, last_step = self.last_reference(block.block_id)
            block_hash: ExternalBlockHash | None = None
            num_tokens: int | None = None
            if block.block_hash is not None:
                block_hash = maybe_convert_block_hash(get_block_hash(block.block_hash))
                num_tokens = block.block_hash_num_tokens
            snapshots.append(
                EvictedBlock(
                    block_id=block.block_id,
                    group_id=group_id,
                    block_hash=block_hash,
                    num_tokens=num_tokens,
                    last_request_id=last_req,
                    last_access_step=last_step,
                    ref_cnt=block.ref_cnt,
                )
            )
        if explanation is None:
            explanation = self._default_explanation(reason, snapshots)
        record = EvictionRecord(
            reason=reason,
            step=self._step,
            blocks=tuple(snapshots),
            explanation=explanation,
        )
        self._history.append(record)
        return record

    @staticmethod
    def _default_explanation(
        reason: EvictionReason, blocks: list[EvictedBlock]
    ) -> str:
        ids = ", ".join(str(b.block_id) for b in blocks)
        if reason is EvictionReason.POOL_REUSE:
            return (
                f"Evicted {len(blocks)} cached block(s) [{ids}] from the prefix "
                "cache because the block pool reused them for a new allocation "
                "(LRU eviction under capacity pressure)."
            )
        if reason is EvictionReason.EXPLICIT:
            return (
                f"Evicted {len(blocks)} cached block(s) [{ids}] from the prefix "
                "cache by explicit request (evict_blocks)."
            )
        if reason is EvictionReason.RESET:
            return (
                f"Evicted {len(blocks)} cached block(s) [{ids}] from the prefix "
                "cache because the prefix cache was reset."
            )
        return (
            f"Replaced hash entries for {len(blocks)} block(s) [{ids}] due to a "
            "partial-to-full cache promotion."
        )

    @property
    def history(self) -> tuple[EvictionRecord, ...]:
        """Read-only view of recorded evictions, oldest first."""
        return tuple(self._history)

    def explain(self) -> str:
        """Return a human-readable explanation of recent evictions."""
        if not self._history:
            return "No KV-cache evictions recorded."
        lines = [f"KV-cache eviction history ({len(self._history)} record(s)):"]
        for record in self._history:
            lines.append(
                f"  step {record.step} [{record.reason.value}]: {record.explanation}"
            )
            for b in record.blocks:
                if b.last_request_id is not None:
                    last = f"last_ref={b.last_request_id}@{b.last_access_step}"
                else:
                    last = "last_ref=<none>"
                lines.append(
                    f"    - block {b.block_id} group={b.group_id} "
                    f"hash={b.block_hash} num_tokens={b.num_tokens} "
                    f"ref_cnt={b.ref_cnt} {last}"
                )
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear the eviction history (does not affect the block pool)."""
        self._history.clear()
