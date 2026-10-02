# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for stream termination-reason reporting.

Covers the ``with_stream_termination`` wrapper directly and, through a raw
ASGI test fixture driving the real FastAPI routers, the full request path of
`/v1/chat/completions` and `/v1/completions`:

* normal completion   -> terminal chunk ``termination_reason: "stop"``;
* client disconnect   -> reason "client_cancelled", no extra tokens sent,
  upstream generator closed (resources released);
* server-side timeout -> terminal chunk ``termination_reason:
  "server_timeout"`` followed by ``[DONE]``, upstream generator closed.

The default (opt-out) path must be byte-identical to the legacy schema.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from vllm.entrypoints.openai.chat_completion.api_router import (
    attach_router as attach_chat_router,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.completion.api_router import (
    attach_router as attach_completion_router,
)
from vllm.entrypoints.openai.completion.protocol import CompletionStreamResponse
from vllm.entrypoints.serve.utils.stream_termination import (
    SSE_DONE_MESSAGE,
    TERMINATION_REASON_CLIENT_CANCELLED,
    TERMINATION_REASON_SERVER_TIMEOUT,
    TERMINATION_REASON_STOP,
    StreamTerminationState,
    with_stream_termination,
)

# Pure CPU/asyncio tests: no engine, no distributed state to clean up.
pytestmark = pytest.mark.skip_global_cleanup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CHUNK_ID = "chatcmpl-test"
CREATED = 1700000000
MODEL = "fake-model"


def chat_chunk(text: str) -> str:
    payload = {
        "id": CHUNK_ID,
        "object": "chat.completion.chunk",
        "created": CREATED,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def completion_chunk(text: str) -> str:
    payload = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": CREATED,
        "model": MODEL,
        "choices": [{"index": 0, "text": text, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def parse_data_frame(frame: str) -> dict:
    assert frame.startswith("data: "), frame
    return json.loads(frame[len("data: ") :])


class ControlledStream:
    """A fake serving-layer SSE generator with observable cleanup."""

    def __init__(self, chunks: list[str], *, block_after: int | None = None):
        self._chunks = chunks
        self._block_after = block_after
        self.release = asyncio.Event()
        self.closed = False
        self.yielded = 0

    async def __call__(self):
        try:
            for i, chunk in enumerate(self._chunks):
                if self._block_after is not None and i == self._block_after:
                    await self.release.wait()
                self.yielded += 1
                yield chunk
        finally:
            self.closed = True


class FakeHandler:
    """Stands in for OpenAIServingChat / OpenAIServingCompletion."""

    def __init__(self, stream_factory):
        self._stream_factory = stream_factory
        self.captured_request = None

    async def create_chat_completion(self, request, raw_request):
        self.captured_request = raw_request
        return self._stream_factory()

    async def create_completion(self, request, raw_request):
        self.captured_request = raw_request
        return self._stream_factory()


def make_chat_app(handler, *, stream_timeout: float = 0) -> FastAPI:
    app = FastAPI()
    attach_chat_router(app)
    app.state.openai_serving_chat = handler
    app.state.args = SimpleNamespace(
        sse_keep_alive_interval=0, stream_timeout=stream_timeout
    )
    return app


def make_completion_app(handler, *, stream_timeout: float = 0) -> FastAPI:
    app = FastAPI()
    attach_completion_router(app)
    app.state.openai_serving_completion = handler
    app.state.args = SimpleNamespace(
        sse_keep_alive_interval=0, stream_timeout=stream_timeout
    )
    return app


async def run_asgi_request(
    app: FastAPI,
    payload: dict,
    *,
    path: str,
    disconnect_after_body_chunks: int | None = None,
) -> list[dict]:
    """Drive the ASGI app directly (no server, no Docker).

    When ``disconnect_after_body_chunks`` is given, an ``http.disconnect``
    message is injected once the app has sent that many response body
    messages, simulating a client going away mid-stream.
    """
    body = json.dumps(payload).encode()
    sent: list[dict] = []
    receive_queue: asyncio.Queue = asyncio.Queue()
    receive_queue.put_nowait({"type": "http.request", "body": body, "more_body": False})
    body_chunks_seen = 0

    async def receive():
        return await receive_queue.get()

    async def send(message):
        nonlocal body_chunks_seen
        sent.append(message)
        if (
            message["type"] == "http.response.body"
            and disconnect_after_body_chunks is not None
        ):
            body_chunks_seen += 1
            if body_chunks_seen == disconnect_after_body_chunks:
                receive_queue.put_nowait({"type": "http.disconnect"})

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=30)
    return sent


def body_frames(sent: list[dict]) -> list[str]:
    frames = []
    for message in sent:
        if message["type"] == "http.response.body":
            frames.append(message["body"].decode())
    return frames


# ---------------------------------------------------------------------------
# Unit tests for with_stream_termination
# ---------------------------------------------------------------------------


def test_disabled_returns_same_generator():
    """With no opt-in and no timeout the exact same object is returned, so
    the default path has zero overhead and the schema cannot change."""

    async def gen():
        yield "x"

    for kwargs in (
        {},
        {"timeout": None},
        {"timeout": 0},
        {"timeout": -1.0},
        {"include_termination_reason": False},
    ):
        g = gen()
        assert with_stream_termination(g, **kwargs) is g


@pytest.mark.asyncio
async def test_normal_completion_terminal_frame_before_done():
    chunks = [chat_chunk("Hello"), chat_chunk(" world"), SSE_DONE_MESSAGE]
    stream = ControlledStream(chunks)
    state = StreamTerminationState()

    out = [
        item
        async for item in with_stream_termination(
            stream(), state=state, include_termination_reason=True
        )
    ]

    # token chunks pass through untouched, then the terminal frame, then
    # [DONE] last.
    assert out[0] == chunks[0]
    assert out[1] == chunks[1]
    terminal = parse_data_frame(out[2])
    assert terminal["termination_reason"] == TERMINATION_REASON_STOP
    # The terminal chunk reuses the stream envelope so it validates against
    # the regular chunk schema.
    assert terminal["id"] == CHUNK_ID
    assert terminal["model"] == MODEL
    assert terminal["object"] == "chat.completion.chunk"
    assert terminal["created"] == CREATED
    assert terminal["choices"] == []
    ChatCompletionStreamResponse.model_validate(terminal)
    assert out[3] == SSE_DONE_MESSAGE
    assert len(out) == 4

    assert state.reason == TERMINATION_REASON_STOP
    assert state.finished
    assert state.upstream_closed.is_set()
    assert stream.closed


@pytest.mark.asyncio
async def test_server_timeout_closes_upstream_and_ends_stream():
    # The upstream generator blocks forever after the first chunk.
    stream = ControlledStream([chat_chunk("Hello"), chat_chunk("never")], block_after=1)
    state = StreamTerminationState()

    out = [
        item
        async for item in with_stream_termination(
            stream(),
            state=state,
            timeout=0.05,
            include_termination_reason=True,
        )
    ]

    assert out[0] == chat_chunk("Hello")
    terminal = parse_data_frame(out[1])
    assert terminal["termination_reason"] == TERMINATION_REASON_SERVER_TIMEOUT
    assert terminal["id"] == CHUNK_ID
    assert out[2] == SSE_DONE_MESSAGE
    assert len(out) == 3

    assert state.reason == TERMINATION_REASON_SERVER_TIMEOUT
    assert state.finished
    # Resources released: the blocked upstream generator was closed.
    assert state.upstream_closed.is_set()
    assert stream.closed
    # The second token was never produced/forwarded.
    assert stream.yielded == 1


@pytest.mark.asyncio
async def test_client_cancellation_records_reason_and_sends_nothing_more():
    stream = ControlledStream(
        [chat_chunk("Hello"), chat_chunk("extra"), SSE_DONE_MESSAGE],
        block_after=1,
    )
    state = StreamTerminationState()
    wrapped = with_stream_termination(
        stream(), state=state, include_termination_reason=True
    )

    received = []

    async def consume():
        async for item in wrapped:
            received.append(item)

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the first chunk through
    while not received:
        await asyncio.sleep(0)
    # Simulate the client going away: cancel the consumer (this is what
    # starlette does to the body iterator on http.disconnect).
    consumer.cancel()
    stream.release.set()  # upstream would produce more tokens now
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert received == [chat_chunk("Hello")]  # no extra tokens after cancel
    assert state.reason == TERMINATION_REASON_CLIENT_CANCELLED
    assert state.finished
    assert state.upstream_closed.is_set()
    assert stream.closed


@pytest.mark.asyncio
async def test_aclose_records_client_cancellation():
    stream = ControlledStream([chat_chunk("Hello"), chat_chunk("extra")], block_after=1)
    state = StreamTerminationState()
    wrapped = with_stream_termination(
        stream(), state=state, include_termination_reason=True
    )

    first = await anext(wrapped)
    assert first == chat_chunk("Hello")
    # aclose() throws GeneratorExit into the wrapper, the other path
    # starlette can take to tear the stream down.
    await wrapped.aclose()
    stream.release.set()

    assert state.reason == TERMINATION_REASON_CLIENT_CANCELLED
    assert state.finished
    assert state.upstream_closed.is_set()
    assert stream.closed


@pytest.mark.asyncio
async def test_timeout_without_opt_in_still_ends_stream_but_schema_unchanged():
    stream = ControlledStream([chat_chunk("Hello"), chat_chunk("never")], block_after=1)
    state = StreamTerminationState()

    out = [
        item
        async for item in with_stream_termination(stream(), state=state, timeout=0.05)
    ]

    # No termination_reason field anywhere (schema unchanged); the stream is
    # still terminated cleanly with [DONE].
    assert out == [chat_chunk("Hello"), SSE_DONE_MESSAGE]
    assert all("termination_reason" not in frame for frame in out)
    assert state.reason == TERMINATION_REASON_SERVER_TIMEOUT
    assert stream.closed


# ---------------------------------------------------------------------------
# ASGI-level tests through the real FastAPI routers
# ---------------------------------------------------------------------------

CHAT_PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "hi"}],
    "stream": True,
    "stream_options": {"include_termination_reason": True},
}

COMPLETION_PAYLOAD = {
    "model": MODEL,
    "prompt": "hello",
    "stream": True,
    "stream_options": {"include_termination_reason": True},
}


@pytest.mark.asyncio
async def test_asgi_chat_normal_completion_terminal_frame():
    chunks = [chat_chunk("Hello"), chat_chunk(" world"), SSE_DONE_MESSAGE]
    stream = ControlledStream(chunks)
    handler = FakeHandler(stream)
    app = make_chat_app(handler)

    sent = await run_asgi_request(app, CHAT_PAYLOAD, path="/v1/chat/completions")

    frames = body_frames(sent)
    # Last ASGI body message is the empty more_body=False terminator.
    assert frames[-1] == ""
    frames = frames[:-1]
    assert frames[0] == chunks[0]
    assert frames[1] == chunks[1]
    terminal = parse_data_frame(frames[2])
    assert terminal["termination_reason"] == TERMINATION_REASON_STOP
    assert terminal["id"] == CHUNK_ID
    assert terminal["model"] == MODEL
    ChatCompletionStreamResponse.model_validate(terminal)
    assert frames[3] == SSE_DONE_MESSAGE

    # The termination state is observable on the request.
    state = handler.captured_request.state.stream_termination
    assert state.reason == TERMINATION_REASON_STOP
    assert state.finished
    assert stream.closed


@pytest.mark.asyncio
async def test_asgi_chat_default_schema_unchanged():
    """Without the opt-in, the byte stream is exactly what the serving layer
    produced: no termination_reason field, no extra frames."""
    chunks = [chat_chunk("Hello"), chat_chunk(" world"), SSE_DONE_MESSAGE]
    stream = ControlledStream(chunks)
    handler = FakeHandler(stream)
    app = make_chat_app(handler)

    payload = {k: v for k, v in CHAT_PAYLOAD.items() if k != "stream_options"}
    sent = await run_asgi_request(app, payload, path="/v1/chat/completions")

    frames = body_frames(sent)
    assert frames[-1] == ""
    assert frames[:-1] == chunks
    assert all("termination_reason" not in frame for frame in frames)


@pytest.mark.asyncio
async def test_asgi_chat_client_disconnect_no_extra_tokens_and_cleanup():
    # The upstream generator blocks after the first token and would produce
    # two more tokens plus [DONE] once released.
    chunks = [
        chat_chunk("Hello"),
        chat_chunk(" world"),
        chat_chunk("!"),
        SSE_DONE_MESSAGE,
    ]
    stream = ControlledStream(chunks, block_after=1)
    handler = FakeHandler(stream)
    app = make_chat_app(handler)

    sent = await run_asgi_request(
        app,
        CHAT_PAYLOAD,
        path="/v1/chat/completions",
        disconnect_after_body_chunks=1,
    )
    # Even though the client is gone, let the upstream generator "produce"
    # its remaining tokens to prove none of them are sent any more.
    stream.release.set()
    await asyncio.sleep(0.05)

    frames = body_frames(sent)
    # Only the first token chunk was ever sent; after the disconnect not a
    # single further byte (no extra tokens, no terminal frame, no [DONE]).
    assert frames == [chunks[0]]

    state = handler.captured_request.state.stream_termination
    assert state.reason == TERMINATION_REASON_CLIENT_CANCELLED
    assert state.finished
    # Resources released: the upstream generator was closed deterministically.
    assert state.upstream_closed.is_set()
    assert stream.closed


@pytest.mark.asyncio
async def test_asgi_chat_server_timeout_terminal_frame():
    chunks = [chat_chunk("Hello"), chat_chunk("never"), SSE_DONE_MESSAGE]
    stream = ControlledStream(chunks, block_after=1)
    handler = FakeHandler(stream)
    app = make_chat_app(handler, stream_timeout=0.05)

    sent = await run_asgi_request(app, CHAT_PAYLOAD, path="/v1/chat/completions")
    stream.release.set()

    frames = body_frames(sent)
    assert frames[-1] == ""
    frames = frames[:-1]
    assert frames[0] == chunks[0]
    terminal = parse_data_frame(frames[1])
    assert terminal["termination_reason"] == TERMINATION_REASON_SERVER_TIMEOUT
    assert terminal["id"] == CHUNK_ID
    ChatCompletionStreamResponse.model_validate(terminal)
    assert frames[2] == SSE_DONE_MESSAGE
    assert len(frames) == 3

    state = handler.captured_request.state.stream_termination
    assert state.reason == TERMINATION_REASON_SERVER_TIMEOUT
    assert state.finished
    assert stream.closed
    assert stream.yielded == 1


@pytest.mark.asyncio
async def test_asgi_completion_normal_completion_terminal_frame():
    chunks = [completion_chunk("Hello"), SSE_DONE_MESSAGE]
    stream = ControlledStream(chunks)
    handler = FakeHandler(stream)
    app = make_completion_app(handler)

    sent = await run_asgi_request(app, COMPLETION_PAYLOAD, path="/v1/completions")

    frames = body_frames(sent)
    assert frames[-1] == ""
    frames = frames[:-1]
    assert frames[0] == chunks[0]
    terminal = parse_data_frame(frames[1])
    assert terminal["termination_reason"] == TERMINATION_REASON_STOP
    assert terminal["id"] == "cmpl-test"
    assert terminal["object"] == "text_completion"
    CompletionStreamResponse.model_validate(terminal)
    assert frames[2] == SSE_DONE_MESSAGE

    state = handler.captured_request.state.stream_termination
    assert state.reason == TERMINATION_REASON_STOP
    assert stream.closed


@pytest.mark.asyncio
async def test_asgi_completion_client_disconnect():
    chunks = [completion_chunk("Hello"), completion_chunk("extra")]
    stream = ControlledStream(chunks, block_after=1)
    handler = FakeHandler(stream)
    app = make_completion_app(handler)

    sent = await run_asgi_request(
        app,
        COMPLETION_PAYLOAD,
        path="/v1/completions",
        disconnect_after_body_chunks=1,
    )
    stream.release.set()
    await asyncio.sleep(0.05)

    assert body_frames(sent) == [chunks[0]]
    state = handler.captured_request.state.stream_termination
    assert state.reason == TERMINATION_REASON_CLIENT_CANCELLED
    assert stream.closed
