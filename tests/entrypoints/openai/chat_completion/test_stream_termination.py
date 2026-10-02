# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproducible ASGI tests for streaming response termination reasons.

These tests exercise the full streaming path (real stream generator + real
termination wrapper + real TerminationStreamingResponse) against a fake
engine client, using an ASGI harness that can disconnect the client or let the
server-side deadline expire.

Run locally with::

    pytest tests/entrypoints/openai/chat_completion/test_stream_termination.py -v
"""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.serve.utils.sse_keep_alive import with_sse_keep_alive
from vllm.entrypoints.serve.utils.stream_termination import (
    TERMINATION_CLIENT_CANCELLED,
    TERMINATION_SERVER_TIMEOUT,
    TERMINATION_STOP,
    TerminationStreamingResponse,
    with_stream_termination,
)
from vllm.outputs import CompletionOutput, RequestOutput


class FakeEngine:
    """A minimal engine client that yields tokens on demand and records aborts."""

    def __init__(self, *, token_delay: float = 0.0, num_tokens: int = 3):
        self.token_delay = token_delay
        self.num_tokens = num_tokens
        self.aborts: list[str] = []
        self.generate_closed = False

    async def generate(self, *args: Any, **kwargs: Any):
        try:
            for i in range(self.num_tokens):
                if self.token_delay:
                    await asyncio.sleep(self.token_delay)
                yield RequestOutput(
                    request_id="test-request",
                    prompt="Hi",
                    prompt_token_ids=[1, 2],
                    prompt_logprobs=None,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text=f"tok{i}",
                            token_ids=[100 + i],
                            cumulative_logprob=None,
                            logprobs=None,
                            finish_reason="stop" if i == self.num_tokens - 1 else None,
                        )
                    ],
                    finished=i == self.num_tokens - 1,
                )
        except (asyncio.CancelledError, GeneratorExit):
            # Mirror the real engine: abort the request when the generator is
            # closed so resources are released promptly.
            await self.abort("test-request", internal=True)
            raise
        finally:
            self.generate_closed = True

    async def abort(self, request_id: str, internal: bool = False) -> None:
        self.aborts.append(request_id)


class FakeServingChat(OpenAIServingChat):
    """A chat serving that builds the real stream generator from a fake engine."""

    def __init__(self, engine: FakeEngine):
        self.engine_client = engine
        self.parser_cls = None
        self.response_role = "assistant"
        self._include_reasoning_tokens_details = False
        self.enable_force_include_usage = False
        self.system_fingerprint = None
        self.enable_per_request_metrics = False
        self.enable_log_outputs = False
        self.enable_log_deltas = False
        self.request_logger = None
        self.enable_prompt_tokens_details = False
        self.enable_auto_tools = False
        self.return_tokens_as_token_ids = False

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request | None = None
    ):
        result_generator = self.engine_client.generate()
        return self.chat_completion_stream_generator(
            request,
            result_generator,
            "chatcmpl-test",
            "test-model",
            conversation=[{"role": "user", "content": "Hi"}],
            tokenizer=MagicMock(),
            request_metadata=RequestResponseMetadata(request_id="chatcmpl-test"),
        )


def _make_scope() -> dict[str, Any]:
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", b"2"),
        ],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
        "scheme": "http",
        "http_version": "1.1",
        "state": {},
    }


async def _run_asgi(app: FastAPI, body: bytes, on_chunk=None):
    """Run the ASGI app, collecting response body chunks.

    ``on_chunk`` is an optional async callable ``(chunk_count, receive_queue)``
    invoked after each body chunk; it may put ``http.disconnect`` on the queue.
    """
    return await _run_asgi_path(app, "/v1/chat/completions", body, on_chunk)


async def _run_asgi_path(
    app: FastAPI, path: str, body: bytes, on_chunk=None
):
    """Run the ASGI app at ``path``, collecting response body chunks."""
    send_queue: asyncio.Queue = asyncio.Queue()
    receive_queue: asyncio.Queue = asyncio.Queue()

    async def receive():
        return await receive_queue.get()

    async def send(message):
        await send_queue.put(message)

    scope = _make_scope()
    scope["path"] = path
    scope["app"] = app

    app_task = asyncio.create_task(app(scope, receive, send))
    await receive_queue.put(
        {"type": "http.request", "body": body, "more_body": False}
    )

    chunks: list[bytes] = []
    while True:
        try:
            message = await asyncio.wait_for(send_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            if app_task.done():
                break
            continue
        if message["type"] == "http.response.body":
            if message["body"]:
                chunks.append(message["body"])
                if on_chunk is not None:
                    await on_chunk(len(chunks), receive_queue)
            if not message.get("more_body", False):
                break

    await app_task
    return chunks, scope


def _build_app(engine: FakeEngine, timeout: float = 0.0) -> FastAPI:
    app = FastAPI()
    app.state.openai_serving_chat = FakeServingChat(engine)

    class _Args:
        sse_keep_alive_interval = 0
        stream_termination_timeout = timeout

    app.state.args = _Args()

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        request: ChatCompletionRequest, raw_request: Request
    ):
        handler = raw_request.app.state.openai_serving_chat
        generator = await handler.create_chat_completion(request, raw_request)
        include_reason = bool(
            request.stream_options
            and request.stream_options.include_termination_reason
        )
        generator = with_stream_termination(
            with_sse_keep_alive(generator, 0.0),
            include_termination_reason=include_reason,
            timeout=float(raw_request.app.state.args.stream_termination_timeout),
            on_termination=lambda reason: setattr(
                raw_request.state, "termination_reason", reason
            ),
        )
        return TerminationStreamingResponse(
            content=generator, media_type="text/event-stream"
        )

    return app


def _parse_frames(chunks: list[bytes]) -> list[dict[str, Any]]:
    """Parse SSE frames (``data: ...``) into a list of payloads."""
    frames: list[dict[str, Any]] = []
    for chunk in chunks:
        for line in chunk.decode().splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                frames.append({"done": True})
            else:
                frames.append(json.loads(payload))
    return frames


async def _collect(generator):
    return [chunk async for chunk in generator]


async def _wait_for_abort(engine: FakeEngine, timeout: float = 2.0) -> None:
    """Poll until the engine's abort is called (async gen cleanup is scheduled
    on the event loop and may run after the ASGI app task completes)."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while not engine.aborts and loop.time() < deadline:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_wrapper_normal_stop():
    """Normal completion emits a stop frame before [DONE]."""

    async def upstream():
        yield "data: {}\n\n"
        yield "data: [DONE]\n\n"

    gen = with_stream_termination(
        upstream(), include_termination_reason=True, timeout=0
    )
    chunks = await _collect(gen)
    assert chunks == [
        "data: {}\n\n",
        'data: {"termination_reason": "stop"}\n\n',
        "data: [DONE]\n\n",
    ]


@pytest.mark.asyncio
async def test_wrapper_default_unchanged():
    """Without opt-in the stream is byte-for-byte unchanged."""

    async def upstream():
        yield "data: {}\n\n"
        yield "data: [DONE]\n\n"

    gen = with_stream_termination(upstream(), include_termination_reason=False)
    chunks = await _collect(gen)
    assert chunks == ["data: {}\n\n", "data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_wrapper_client_cancelled():
    """Client cancellation records the reason and closes the upstream."""
    closed = False

    async def upstream():
        nonlocal closed
        try:
            yield "data: {}\n\n"
            await asyncio.sleep(100)
        finally:
            closed = True

    reasons: list[str] = []
    gen = with_stream_termination(
        upstream(),
        include_termination_reason=True,
        on_termination=reasons.append,
    )
    await gen.__anext__()
    task = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await gen.aclose()
    assert reasons == [TERMINATION_CLIENT_CANCELLED]
    assert closed


@pytest.mark.asyncio
async def test_wrapper_server_timeout():
    """Server timeout emits a server_timeout frame and closes the upstream."""
    closed = False

    async def upstream():
        nonlocal closed
        try:
            yield "data: {}\n\n"
            await asyncio.sleep(100)
        finally:
            closed = True

    reasons: list[str] = []
    gen = with_stream_termination(
        upstream(),
        include_termination_reason=True,
        timeout=0.1,
        on_termination=reasons.append,
    )
    chunks = []
    with pytest.raises(StopAsyncIteration):
        while True:
            chunks.append(await gen.__anext__())
    assert 'data: {"termination_reason": "server_timeout"}\n\n' in chunks
    assert chunks[-1] == "data: [DONE]\n\n"
    assert reasons == [TERMINATION_SERVER_TIMEOUT]
    assert closed


@pytest.mark.asyncio
async def test_wrapper_normal_stop_with_timeout_that_does_not_fire():
    """Stream completes before the deadline: stop frame, no timeout, no loop."""

    async def upstream():
        yield "data: {}\n\n"
        yield "data: [DONE]\n\n"

    reasons: list[str] = []
    gen = with_stream_termination(
        upstream(),
        include_termination_reason=True,
        timeout=10.0,  # deadline far in the future
        on_termination=reasons.append,
    )
    chunks = []
    with pytest.raises(StopAsyncIteration):
        while True:
            chunks.append(await gen.__anext__())
    assert chunks == [
        "data: {}\n\n",
        'data: {"termination_reason": "stop"}\n\n',
        "data: [DONE]\n\n",
    ]
    assert reasons == [TERMINATION_STOP]


def _request_body(*, include_reason: bool = True) -> bytes:
    return json.dumps(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "stream_options": {"include_termination_reason": include_reason},
        }
    ).encode()


@pytest.mark.asyncio
async def test_asgi_normal_stop():
    """Normal stream ends with a stop frame then [DONE]; no abort."""
    engine = FakeEngine(token_delay=0.0, num_tokens=3)
    app = _build_app(engine, timeout=0.0)

    chunks, scope = await _run_asgi(app, _request_body())
    frames = _parse_frames(chunks)

    assert frames[-2] == {"termination_reason": TERMINATION_STOP}
    assert frames[-1] == {"done": True}
    assert engine.aborts == []
    assert scope["state"]["termination_reason"] == TERMINATION_STOP


@pytest.mark.asyncio
async def test_asgi_default_schema_unchanged():
    """Without opt-in no termination frame is emitted."""
    engine = FakeEngine(token_delay=0.0, num_tokens=2)
    app = _build_app(engine, timeout=0.0)

    chunks, _ = await _run_asgi(app, _request_body(include_reason=False))
    frames = _parse_frames(chunks)

    assert all("termination_reason" not in f for f in frames if "done" not in f)
    assert frames[-1] == {"done": True}


@pytest.mark.asyncio
async def test_asgi_client_disconnect():
    """Client disconnect stops the stream, aborts the engine, records reason."""
    engine = FakeEngine(token_delay=0.1, num_tokens=10)
    app = _build_app(engine, timeout=0.0)

    async def disconnect_after_2(count, receive_queue):
        if count == 2:
            await receive_queue.put({"type": "http.disconnect"})

    chunks, scope = await _run_asgi(app, _request_body(), on_chunk=disconnect_after_2)
    frames = _parse_frames(chunks)

    content_frames = [f for f in frames if "done" not in f]
    assert len(content_frames) == 2
    assert not any(f.get("done") for f in frames)
    await _wait_for_abort(engine)
    assert engine.aborts, "engine abort should be called on client disconnect"
    assert scope["state"]["termination_reason"] == TERMINATION_CLIENT_CANCELLED


@pytest.mark.asyncio
async def test_asgi_server_timeout():
    """Server deadline emits a server_timeout frame then [DONE], aborts engine."""
    engine = FakeEngine(token_delay=1.0, num_tokens=3)
    app = _build_app(engine, timeout=0.3)

    chunks, scope = await _run_asgi(app, _request_body())
    frames = _parse_frames(chunks)

    assert frames[-2] == {"termination_reason": TERMINATION_SERVER_TIMEOUT}
    assert frames[-1] == {"done": True}
    await _wait_for_abort(engine)
    assert engine.aborts, "engine abort should be called on server timeout"
    assert scope["state"]["termination_reason"] == TERMINATION_SERVER_TIMEOUT


# ---------------------------------------------------------------------------
# Completion endpoint test
# ---------------------------------------------------------------------------


class FakeServingCompletion(OpenAIServingCompletion):
    """A completion serving that builds the real stream generator from a fake engine."""

    def __init__(self, engine: FakeEngine):
        self.engine_client = engine
        self.enable_force_include_usage = False
        self.system_fingerprint = None
        self.enable_per_request_metrics = False
        self.enable_prompt_tokens_details = False
        self.return_tokens_as_token_ids = False

    async def create_completion(
        self, request: CompletionRequest, raw_request: Request | None = None
    ):
        from vllm.utils.async_utils import merge_async_iterators

        generators = [self.engine_client.generate()]
        result_generator = merge_async_iterators(*generators)
        return self.completion_stream_generator(
            request,
            [],  # engine_inputs (unused since res.prompt is set)
            result_generator,
            "cmpl-test",
            0,
            "test-model",
            1,
            MagicMock(),
            RequestResponseMetadata(request_id="cmpl-test"),
        )


def _build_completion_app(engine: FakeEngine, timeout: float = 0.0) -> FastAPI:
    app = FastAPI()
    app.state.openai_serving_completion = FakeServingCompletion(engine)

    class _Args:
        sse_keep_alive_interval = 0
        stream_termination_timeout = timeout

    app.state.args = _Args()

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest, raw_request: Request):
        handler = raw_request.app.state.openai_serving_completion
        generator = await handler.create_completion(request, raw_request)
        include_reason = bool(
            request.stream_options
            and request.stream_options.include_termination_reason
        )
        generator = with_stream_termination(
            with_sse_keep_alive(generator, 0.0),
            include_termination_reason=include_reason,
            timeout=float(raw_request.app.state.args.stream_termination_timeout),
            on_termination=lambda reason: setattr(
                raw_request.state, "termination_reason", reason
            ),
        )
        return TerminationStreamingResponse(
            content=generator, media_type="text/event-stream"
        )

    return app


def _completion_request_body(*, include_reason: bool = True) -> bytes:
    return json.dumps(
        {
            "model": "test-model",
            "prompt": "Hi",
            "stream": True,
            "stream_options": {"include_termination_reason": include_reason},
        }
    ).encode()


@pytest.mark.asyncio
async def test_completion_normal_stop():
    """Completion stream ends with a stop frame then [DONE]; no abort."""
    engine = FakeEngine(token_delay=0.0, num_tokens=3)
    app = _build_completion_app(engine, timeout=0.0)

    chunks, scope = await _run_asgi_path(
        app, "/v1/completions", _completion_request_body()
    )
    frames = _parse_frames(chunks)

    assert frames[-2] == {"termination_reason": TERMINATION_STOP}
    assert frames[-1] == {"done": True}
    assert engine.aborts == []
    assert scope["state"]["termination_reason"] == TERMINATION_STOP


@pytest.mark.asyncio
async def test_completion_server_timeout():
    """Completion stream timeout emits a server_timeout frame then [DONE]."""
    engine = FakeEngine(token_delay=1.0, num_tokens=3)
    app = _build_completion_app(engine, timeout=0.3)

    chunks, scope = await _run_asgi_path(
        app, "/v1/completions", _completion_request_body()
    )
    frames = _parse_frames(chunks)

    assert frames[-2] == {"termination_reason": TERMINATION_SERVER_TIMEOUT}
    assert frames[-1] == {"done": True}
    await _wait_for_abort(engine)
    assert engine.aborts, "engine abort should be called on server timeout"
    assert scope["state"]["termination_reason"] == TERMINATION_SERVER_TIMEOUT
