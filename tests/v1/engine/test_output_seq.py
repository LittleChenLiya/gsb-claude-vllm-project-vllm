# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Acceptance tests for output deduplication sequence numbers.

A fake engine injects:

* **duplicate** EngineCoreOutput events (the same engine step is redelivered,
  emulating a replay after a client reconnect),
* **out-of-order** processing of events belonging to different requests
  (requests are interleaved in arbitrary order),
* **completion** events (a final output carrying ``finished=True``).

Every emitted ``RequestOutput`` must carry a strictly increasing, per-request,
1-based ``output_seq`` which a reconnecting client can use for deduplication.
"""

import pytest

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    RequestOutputCollector,
)


def _make_request(
    internal_id: str,
    external_id: str,
    prompt_token_ids: list[int] | None = None,
) -> EngineCoreRequest:
    # detokenize=False keeps the test tokenizer-free (no model download).
    return EngineCoreRequest(
        request_id=internal_id,
        external_req_id=external_id,
        prompt_token_ids=prompt_token_ids
        if prompt_token_ids is not None
        else [1, 2, 3],
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


def _token_event(request_id: str, token_id: int) -> EngineCoreOutput:
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[token_id],
    )


def _finish_event(request_id: str) -> EngineCoreOutput:
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[999],
        finish_reason=FinishReason.LENGTH,
    )


@pytest.fixture()
def processor() -> OutputProcessor:
    return OutputProcessor(tokenizer=None, log_stats=False, stream_interval=1)


def _add(
    processor: OutputProcessor,
    internal_id: str,
    external_id: str | None = None,
    queue=None,
) -> None:
    processor.add_request(
        _make_request(internal_id, external_id or internal_id.replace("-int", "")),
        prompt=None,
        queue=queue,
    )


def test_sequence_numbers_monotonic_including_completion(processor: OutputProcessor):
    """Each emitted output (incl. the finished one) is numbered 1, 2, 3, ..."""
    _add(processor, "req-A-int", "req-A")

    result = processor.process_outputs(
        [
            _token_event("req-A-int", 10),
            _token_event("req-A-int", 11),
            _token_event("req-A-int", 12),
            _finish_event("req-A-int"),
        ]
    )

    seqs = [o.output_seq for o in result.request_outputs]
    assert seqs == [1, 2, 3, 4]

    finished = result.request_outputs[-1]
    assert finished.finished is True
    assert finished.output_seq == 4
    # Counter is released once the external request fully completes.
    assert "req-A" not in processor.output_seq_counters
    assert not processor.has_unfinished_requests()


def test_duplicate_events_get_distinct_sequence_numbers(processor: OutputProcessor):
    """A redelivered engine event is still a newly *emitted* output, hence gets
    the next sequence number; the emitted numbers themselves never repeat,
    which is what makes a redelivered *output object* detectable downstream.
    """
    _add(processor, "req-A-int", "req-A")

    # Fake engine re-sends the exact same event twice (replay after reconnect).
    result = processor.process_outputs(
        [
            _token_event("req-A-int", 10),
            _token_event("req-A-int", 10),
        ]
    )
    seqs = [o.output_seq for o in result.request_outputs]
    payloads = [o.outputs[0].token_ids for o in result.request_outputs]

    assert seqs == [1, 2]
    assert payloads == [[10], [10]]  # identical payload on the replay
    assert len(seqs) == len(set(seqs))  # numbers are never reused


def test_different_requests_number_independently_even_when_interleaved(
    processor: OutputProcessor,
):
    """Out-of-order / interleaved processing must not couple the counters."""
    _add(processor, "req-A-int", "req-A")
    _add(processor, "req-B-int", "req-B")

    # Interleave the two requests in arbitrary order within one engine batch.
    result = processor.process_outputs(
        [
            _token_event("req-B-int", 20),
            _token_event("req-A-int", 10),
            _token_event("req-B-int", 21),
            _token_event("req-A-int", 11),
            _token_event("req-B-int", 22),
        ]
    )

    by_req: dict[str, list[int]] = {}
    for output in result.request_outputs:
        by_req.setdefault(output.request_id, []).append(output.output_seq)

    assert by_req == {
        "req-B": [1, 2, 3],
        "req-A": [1, 2],
    }

    # Finishing one request must not disturb the other's numbering.
    result = processor.process_outputs(
        [
            _token_event("req-A-int", 12),
            _finish_event("req-A-int"),
            _token_event("req-B-int", 23),
        ]
    )
    seqs_a = [o.output_seq for o in result.request_outputs if o.request_id == "req-A"]
    seqs_b = [o.output_seq for o in result.request_outputs if o.request_id == "req-B"]
    assert seqs_a == [3, 4]
    assert seqs_b == [4]
    assert "req-A" not in processor.output_seq_counters
    assert processor.output_seq_counters["req-B"] == 4


def test_request_reusing_external_id_restarts_at_one(processor: OutputProcessor):
    """After completion the counter is released; a new independent request
    reusing the external id restarts numbering at 1."""
    _add(processor, "req-A-int", "req-A")
    result = processor.process_outputs(
        [_token_event("req-A-int", 10), _finish_event("req-A-int")]
    )
    assert [o.output_seq for o in result.request_outputs] == [1, 2]

    _add(processor, "req-A2-int", "req-A")
    result = processor.process_outputs([_token_event("req-A2-int", 30)])
    assert result.request_outputs[0].output_seq == 1


def test_client_reconnect_dedup_simulation(processor: OutputProcessor):
    """End-to-end client behavior: after reconnect, apply only outputs whose
    sequence number exceeds the last applied number; replays are dropped and
    out-of-order arrivals are buffered until gaps close."""
    _add(processor, "req-A-int", "req-A")

    # First wave the client received and applied: seqs 1 and 2.
    first_wave = processor.process_outputs(
        [
            _token_event("req-A-int", 10),
            _token_event("req-A-int", 11),
        ]
    )
    assert [o.output_seq for o in first_wave.request_outputs] == [1, 2]

    # New deltas seq 3, seq 4 plus the completion seq 5.
    second_wave = processor.process_outputs(
        [
            _token_event("req-A-int", 12),
            _token_event("req-A-int", 13),
            _finish_event("req-A-int"),
        ]
    )
    assert [o.output_seq for o in second_wave.request_outputs] == [3, 4, 5]

    # Reconnected client stream: a redelivered seq 2 (duplicate), then seq 4
    # before seq 3 (out-of-order), then seq 5 (finished).
    redelivered_seq2 = first_wave.request_outputs[1]
    stream = [
        redelivered_seq2,
        second_wave.request_outputs[1],  # seq 4 arrives first
        second_wave.request_outputs[0],  # seq 3
        second_wave.request_outputs[2],  # seq 5 (finished)
    ]

    last_applied = 2  # high-water mark persisted by the client
    buffer: dict[int, object] = {}
    applied: list[int] = []
    for output in stream:
        seq = output.output_seq
        if seq <= last_applied:
            continue  # duplicate/replay after reconnect
        buffer[seq] = output
        while last_applied + 1 in buffer:
            last_applied += 1
            applied.append(last_applied)

    assert applied == [3, 4, 5]
    assert buffer[5].finished is True


def test_abort_output_is_numbered_and_counter_released(processor: OutputProcessor):
    """Abort also produces a numbered terminal output."""
    collector = RequestOutputCollector(
        output_kind=RequestOutputKind.DELTA, request_id="req-A"
    )
    _add(processor, "req-A-int", "req-A", queue=collector)
    processor.process_outputs([_token_event("req-A-int", 10)])
    first = collector.get_nowait()
    assert first.output_seq == 1

    processor.abort_requests(["req-A-int"], internal=True)
    aborted = collector.get_nowait()
    assert aborted.output_seq == 2
    assert aborted.finished is True
    assert "req-A" not in processor.output_seq_counters


def test_abort_by_external_id_is_numbered_and_counter_released(
    processor: OutputProcessor,
):
    """Abort via the external request id also emits a numbered terminal output
    and releases the counter."""
    collector = RequestOutputCollector(
        output_kind=RequestOutputKind.DELTA, request_id="req-A"
    )
    _add(processor, "req-A-int", "req-A", queue=collector)
    processor.process_outputs([_token_event("req-A-int", 10)])
    assert collector.get_nowait().output_seq == 1

    # external (client-facing) id
    processor.abort_requests(["req-A"], internal=False)
    aborted = collector.get_nowait()
    assert aborted.output_seq == 2
    assert aborted.finished is True
    assert "req-A" not in processor.output_seq_counters


def test_collector_merge_keeps_highest_in_either_order():
    """When the producer outruns the consumer the collector coalesces outputs;
    the merged sequence number is the highest seen regardless of merge order,
    so a request's sequence can never go backwards."""
    from vllm.outputs import CompletionOutput, RequestOutput

    def _ro(seq: int) -> RequestOutput:
        return RequestOutput(
            request_id="x",
            prompt=None,
            prompt_token_ids=None,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                    cumulative_logprob=None,
                    logprobs=None,
                )
            ],
            finished=False,
            output_seq=seq,
        )

    # Older merged into newer -> newer number wins.
    newer = _ro(2)
    newer.add(_ro(1), aggregate=True)
    assert newer.output_seq == 2

    # Newer merged into older (producer outran consumer) -> still the high.
    older = _ro(1)
    older.add(_ro(2), aggregate=True)
    assert older.output_seq == 2

    # And the real collector path: two queued deltas coalesce into one object
    # carrying the highest sequence number.
    collector = RequestOutputCollector(
        output_kind=RequestOutputKind.DELTA, request_id="x"
    )
    collector.put(_ro(1))
    collector.put(_ro(2))
    merged = collector.get_nowait()
    assert merged.output_seq == 2


def test_output_processor_rejects_re_stamped_regression(processor: OutputProcessor):
    """The stamping choke point fails loudly if an already-numbered output with
    a smaller number is about to be emitted after a newer one."""
    from vllm.outputs import CompletionOutput, RequestOutput

    _add(processor, "req-A-int", "req-A")
    processor.output_seq_counters["req-A"] = 5

    stale = RequestOutput(
        request_id="req-A",
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[],
                cumulative_logprob=None,
                logprobs=None,
            )
        ],
        finished=False,
        output_seq=3,  # regression: 3 after 5 already emitted
    )
    with pytest.raises(AssertionError, match="regressed"):
        processor._assign_output_seq(stale, "req-A")
