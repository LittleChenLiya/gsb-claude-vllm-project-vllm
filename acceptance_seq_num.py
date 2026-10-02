# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance test for output dedup sequence numbers.

A fake engine injects a stream of EngineCoreOutputs for two requests,
including duplicate and out-of-order events plus a completion event.
The OutputProcessor must:

* assign a strictly increasing per-request seq num to every emitted output;
* drop duplicate engine outputs (same seq num re-delivered);
* drop out-of-order engine outputs (seq num already passed);
* keep the two requests' seq nums independent.
"""

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor


def make_request(internal_id: str, external_id: str) -> EngineCoreRequest:
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
        ),
        pooling_params=None,
    )


def main() -> None:
    processor = OutputProcessor(tokenizer=None, log_stats=False)

    req_a = make_request("A-int", "A")
    req_b = make_request("B-int", "B")
    processor.add_request(req_a, prompt=None)
    processor.add_request(req_b, prompt=None)

    # --- Fake engine event stream -----------------------------------------
    # Request A: normal 1..5, with a duplicate of seq 3 and a duplicate of
    # the completion (seq 5).
    # Request B: seq 1 then seq 3 (out of order) then seq 2 (late, must be
    # dropped) then seq 4 completion.
    events: list[EngineCoreOutput] = [
        # A: normal progression
        EngineCoreOutput("A-int", [101], seq_num=1),
        EngineCoreOutput("A-int", [102], seq_num=2),
        EngineCoreOutput("A-int", [103], seq_num=3),
        # A: duplicate of seq 3 (at-least-once re-delivery)
        EngineCoreOutput("A-int", [103], seq_num=3),
        # B: seq 1 then seq 3 (out-of-order delivery)
        EngineCoreOutput("B-int", [201], seq_num=1),
        EngineCoreOutput("B-int", [203], seq_num=3),
        # B: late seq 2 (must be dropped, seq num already passed)
        EngineCoreOutput("B-int", [202], seq_num=2),
        # A: continue
        EngineCoreOutput("A-int", [104], seq_num=4),
        # B: completion
        EngineCoreOutput(
            "B-int", [204], seq_num=4, finish_reason=FinishReason.LENGTH
        ),
        # A: completion
        EngineCoreOutput(
            "A-int", [105], seq_num=5, finish_reason=FinishReason.LENGTH
        ),
        # A: duplicate of the completion (must be dropped)
        EngineCoreOutput(
            "A-int", [105], seq_num=5, finish_reason=FinishReason.LENGTH
        ),
    ]

    # Feed events one at a time so the emitted seq nums are clearly visible.
    emitted: dict[str, list[tuple[int, bool, list[int]]]] = {"A": [], "B": []}
    for event in events:
        result = processor.process_outputs([event])
        for out in result.request_outputs:
            emitted[out.request_id].append(
                (out.seq_num, out.finished, list(out.outputs[0].token_ids))
            )

    # --- Verify ------------------------------------------------------------
    print("=" * 70)
    print("Emitted outputs (seq_num, finished, token_ids):")
    print("=" * 70)
    for req_id in ("A", "B"):
        print(f"\nRequest {req_id}:")
        for seq_num, finished, token_ids in emitted[req_id]:
            print(
                f"  seq_num={seq_num:>2}  finished={finished!s:<5}  "
                f"token_ids={token_ids}"
            )

    # Expected: A emits 5 outputs (duplicates dropped); B emits 3 (late seq 2
    # dropped).
    expected_a = [
        (1, False, [101]),
        (2, False, [102]),
        (3, False, [103]),
        (4, False, [104]),
        (5, True, [105]),
    ]
    expected_b = [
        (1, False, [201]),
        (2, False, [203]),  # req seq 2 <- engine seq 3 (token 203)
        (3, True, [204]),   # req seq 3 <- engine seq 4 (completion)
    ]

    assert emitted["A"] == expected_a, f"A mismatch: {emitted['A']}"
    assert emitted["B"] == expected_b, f"B mismatch: {emitted['B']}"

    # Strict monotonicity per request (no regression).
    for req_id in ("A", "B"):
        seq_nums = [s for s, _, _ in emitted[req_id]]
        assert seq_nums == sorted(seq_nums), f"{req_id} seq nums not monotonic"
        assert len(set(seq_nums)) == len(seq_nums), f"{req_id} has duplicate seq"

    # Independence: A and B both start at seq 1 and don't share state.
    assert emitted["A"][0][0] == 1 and emitted["B"][0][0] == 1

    # Both requests finished and removed.
    assert processor.get_num_unfinished_requests() == 0

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
    print(
        "\nBehavior summary:\n"
        "  * Duplicate engine outputs (A seq 3, A seq 5) are dropped.\n"
        "  * Out-of-order engine output (B seq 2 after seq 3) is dropped.\n"
        "  * Every emitted output has a strictly increasing per-request seq num.\n"
        "  * Requests A and B maintain independent seq counters.\n"
    )


if __name__ == "__main__":
    main()
