# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Standalone acceptance demo for output deduplication sequence numbers.

Run with::

    python tests/v1/engine/demo_output_seq.py

A fake engine feeds the real ``OutputProcessor`` events that include:

1. duplicate events (same EngineCoreOutput redelivered, as after a reconnect),
2. out-of-order/interleaved events for two different requests,
3. a completion event (``finished=True``).

The script prints every emitted output together with its per-request monotonic
``output_seq`` and then simulates a reconnecting client that deduplicates using
the sequence numbers.
"""

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor


def make_request(internal_id: str, external_id: str) -> EngineCoreRequest:
    # detokenize=False => no tokenizer/model download is required.
    return EngineCoreRequest(
        request_id=internal_id,
        external_req_id=external_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(
            detokenize=False,
            output_kind=RequestOutputKind.DELTA,
            max_tokens=100,
        ),
        pooling_params=None,
    )


def tok(request_id: str, token_id: int, finish: bool = False) -> EngineCoreOutput:
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[token_id],
        finish_reason=FinishReason.LENGTH if finish else None,
    )


def show(label: str, outputs) -> None:
    print(f"--- {label} ---")
    for out in outputs:
        token_ids = [c.token_ids for c in out.outputs]
        print(
            f"  request={out.request_id:<5} output_seq={out.output_seq} "
            f"token_ids={token_ids} finished={out.finished}"
        )


def main() -> None:
    proc = OutputProcessor(tokenizer=None, log_stats=False, stream_interval=1)
    proc.add_request(make_request("A-int", "A"), prompt=None)
    proc.add_request(make_request("B-int", "B"), prompt=None)

    # Batch 1: interleaved requests, out-of-order order B,A,B,A,B.
    batch1 = [
        tok("B-int", 20),
        tok("A-int", 10),
        tok("B-int", 21),
        tok("A-int", 11),
        tok("B-int", 22),
    ]
    out1 = proc.process_outputs(batch1).request_outputs
    show("batch 1: interleaved A/B events", out1)

    # Batch 2: A completes; B continues. Also inject a DUPLICATE of the
    # A-int token 11 event (a redelivered engine event after reconnect).
    duplicate = tok("A-int", 11)
    batch2 = [
        duplicate,
        tok("A-int", 12),
        tok("A-int", 13, finish=True),  # completion event
        tok("B-int", 23),
    ]
    out2 = proc.process_outputs(batch2).request_outputs
    show("batch 2: duplicated A event + more A events + A completion", out2)

    # --- Simulated reconnecting client for request A ----------------------
    # Client already applied seq 1 and seq 2 before dropping the connection.
    # On reconnect the server redelivers seq 2 (same object), then the client
    # receives seq 4 before seq 3 (out-of-order arrival), then seq 5 (final).
    a_wave1 = [o for o in out1 if o.request_id == "A"]  # seq 1, 2
    a_wave2 = [o for o in out2 if o.request_id == "A"]  # dup-emit 3, 4, 5
    stream = [
        a_wave1[1],  # redelivered seq 2 (duplicate)
        a_wave2[1],  # seq 4 arrives early
        a_wave2[0],  # seq 3
        a_wave2[2],  # seq 5 (finished)
    ]

    print("--- reconnect dedup simulation for request A (last_applied=2) ---")
    last_applied = 2
    pending: dict[int, object] = {}
    applied: list[int] = []
    for out in stream:
        seq = out.output_seq
        action = f"seq={seq}"
        if seq <= last_applied:
            action += " -> DROP (duplicate/replay)"
        else:
            pending[seq] = out
            action += " -> accept"
            while last_applied + 1 in pending:
                last_applied += 1
                applied.append(last_applied)
                action += f"; applied {last_applied}"
        print(f"  recv {action}")
    print(f"  applied in order: {applied}")
    print(f"  final finished flag: {pending[5].finished}")

    assert applied == [3, 4, 5]
    assert pending[5].finished is True
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
