# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stream termination reason reporting for SSE streaming responses.

OpenAI-compatible streaming endpoints (`/v1/chat/completions` and
`/v1/completions`) end a stream with ``data: [DONE]`` and no indication of
*why* the stream ended. ``with_stream_termination`` wraps the final SSE
generator and, when the client opted in via
``stream_options.include_termination_reason``, emits one extra terminal
chunk carrying a ``termination_reason`` field right before ``[DONE]``:

* ``"stop"``             - the request finished normally;
* ``"client_cancelled"`` - the client disconnected mid-stream (no further
  frames are sent to the client in this case);
* ``"server_timeout"``   - the server-side stream deadline
  (``--stream-timeout``) was hit; the upstream generator is closed and the
  client still receives the terminal chunk followed by ``[DONE]``.

The wrapper is a no-op (identity) when neither the termination frame nor a
timeout is requested, so the default response schema and the default code
path are completely unchanged.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Final

from vllm.logger import init_logger

logger = init_logger(__name__)

SSE_DONE_MESSAGE: Final = "data: [DONE]\n\n"
_DATA_PREFIX: Final = "data: "

# Termination reasons reported in the terminal chunk / termination state.
TERMINATION_REASON_STOP: Final = "stop"
TERMINATION_REASON_CLIENT_CANCELLED: Final = "client_cancelled"
TERMINATION_REASON_SERVER_TIMEOUT: Final = "server_timeout"

# Envelope fields copied from the first data chunk into the terminal chunk so
# that the terminal chunk validates against the same ``*StreamResponse``
# schema as every other chunk of the stream.
_ENVELOPE_KEYS: Final = ("id", "object", "created", "model")


@dataclass
class StreamTerminationState:
    """Observable outcome of a streaming response.

    The API routers attach an instance to ``request.state.stream_termination``
    so middleware, background tasks and tests can inspect how the stream
    ended. ``reason`` stays ``None`` until the stream terminates.
    """

    reason: str | None = None
    finished: bool = False
    # Set once the upstream generator has been closed (resources released).
    upstream_closed: asyncio.Event = field(default_factory=asyncio.Event)


def _extract_envelope(chunk: str) -> dict[str, Any] | None:
    """Extract the reusable envelope fields from an SSE data chunk."""
    if not chunk.startswith(_DATA_PREFIX):
        return None
    payload = chunk[len(_DATA_PREFIX) :].strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    return {k: data[k] for k in _ENVELOPE_KEYS if k in data}


def _termination_frame(envelope: dict[str, Any] | None, reason: str) -> str:
    """Build the terminal SSE chunk carrying ``termination_reason``."""
    frame: dict[str, Any] = dict(envelope) if envelope else {}
    frame.setdefault("choices", [])
    frame["termination_reason"] = reason
    return f"{_DATA_PREFIX}{json.dumps(frame)}\n\n"


async def _close_upstream(
    generator: AsyncGenerator[str, None],
    next_task: "asyncio.Task[str]",
    state: StreamTerminationState,
) -> None:
    """Cancel the pending advance and close the upstream generator.

    Best-effort: cleanup must never mask the original control flow, and it
    may itself be cancelled (e.g. anyio level-triggered cancellation on
    client disconnect), so all exceptions are suppressed.
    """
    next_task.cancel()
    with contextlib.suppress(BaseException):
        await next_task
    with contextlib.suppress(BaseException):
        await generator.aclose()
    state.upstream_closed.set()


def with_stream_termination(
    generator: AsyncGenerator[str, None],
    *,
    state: StreamTerminationState | None = None,
    timeout: float | None = None,
    include_termination_reason: bool = False,
) -> AsyncGenerator[str, None]:
    """Augment an SSE generator with termination-reason reporting.

    Args:
        generator: The upstream SSE generator (already including keep-alive).
        state: Mutable state object updated with the termination reason.
        timeout: Optional server-side deadline in seconds for the whole
            stream. On expiry the upstream generator is closed and the
            stream ends with a ``"server_timeout"`` terminal chunk (if
            requested) followed by ``[DONE]``.
        include_termination_reason: Emit the extra terminal chunk carrying
            ``termination_reason`` before ``[DONE]``.

    When neither a timeout nor the terminal chunk is requested, the exact
    same generator object is returned (identity), so the default path has no
    overhead and the response schema is unchanged.

    """
    if state is None:
        state = StreamTerminationState()
    if timeout is not None and not 0 < timeout < float("inf"):
        timeout = None
    if not include_termination_reason and timeout is None:
        return generator
    return _terminating_stream(generator, state, timeout, include_termination_reason)


async def _terminating_stream(
    generator: AsyncGenerator[str, None],
    state: StreamTerminationState,
    timeout: float | None,
    include_termination_reason: bool,
) -> AsyncGenerator[str, None]:
    deadline = None if timeout is None else time.monotonic() + timeout
    envelope: dict[str, Any] | None = None
    # A ``data: [DONE]`` frame received from upstream is held back so the
    # termination frame can be emitted *before* it.
    pending_done = False

    # A single retained ``anext()`` task advances the generator; asyncio
    # forbids concurrent ``__anext__`` calls, and ``wait_for`` would cancel
    # the task mid-advance on timeout. ``asyncio.wait`` observes both
    # outcomes without cancelling the pending advance.
    next_task: asyncio.Task[str] = asyncio.create_task(anext(generator))
    try:
        while True:
            wait_timeout = (
                None if deadline is None else max(deadline - time.monotonic(), 0.0)
            )
            done, _ = await asyncio.wait({next_task}, timeout=wait_timeout)
            if not done:
                # Server-side timeout: close the upstream generator (resource
                # release) and end the stream cleanly for the client.
                state.reason = TERMINATION_REASON_SERVER_TIMEOUT
                state.finished = True
                logger.info(
                    "Streaming response terminated by server after %.3fs.",
                    timeout,
                )
                await _close_upstream(generator, next_task, state)
                if include_termination_reason:
                    yield _termination_frame(envelope, state.reason)
                yield SSE_DONE_MESSAGE
                return
            try:
                chunk = next_task.result()
            except StopAsyncIteration:
                break
            if include_termination_reason:
                if chunk == SSE_DONE_MESSAGE:
                    pending_done = True
                elif envelope is None:
                    envelope = _extract_envelope(chunk)
            if not pending_done:
                yield chunk
            next_task = asyncio.create_task(anext(generator))

        # Normal completion.
        state.reason = TERMINATION_REASON_STOP
        state.finished = True
        await _close_upstream(generator, next_task, state)
        if include_termination_reason:
            yield _termination_frame(envelope, state.reason)
        if pending_done:
            yield SSE_DONE_MESSAGE
    except (asyncio.CancelledError, GeneratorExit):
        # Client disconnected (starlette cancels/closes the response body
        # iterator). Record the reason and release the upstream generator,
        # but do NOT emit any further frames: the client is gone and no
        # extra tokens may be sent after a cancellation.
        state.reason = TERMINATION_REASON_CLIENT_CANCELLED
        state.finished = True
        logger.debug("Streaming response cancelled by client disconnect.")
        await _close_upstream(generator, next_task, state)
        raise
    finally:
        if not state.finished:
            # Unexpected exception path: the stream is over either way.
            state.finished = True
        if not next_task.done():
            next_task.cancel()
            with contextlib.suppress(BaseException):
                await next_task
