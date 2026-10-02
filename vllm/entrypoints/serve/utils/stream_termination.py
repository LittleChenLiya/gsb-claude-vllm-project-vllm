# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Termination reason reporting for OpenAI-compatible SSE streams.

Wraps the final SSE generator so callers can observe *why* a stream ended:

- ``stop``: the stream finished normally (every token was emitted).
- ``client_cancelled``: the client disconnected before the stream finished.
- ``server_timeout``: the server-side stream deadline was exceeded.

The termination reason is emitted as a dedicated SSE event
(``data: {"termination_reason": "<reason>"}\n\n``) immediately before the
terminal ``data: [DONE]\n\n`` marker. Emitting it is opt-in via
``stream_options.include_termination_reason`` so the default wire format is
byte-for-byte unchanged.

On client cancellation the wrapper closes the upstream generator, which aborts
the engine request and releases its resources, and re-raises without emitting
any further tokens. On server timeout it emits the terminal frames and closes
the upstream as well.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, Callable

from fastapi.responses import StreamingResponse

from vllm.logger import init_logger

logger = init_logger(__name__)

TERMINATION_STOP = "stop"
TERMINATION_CLIENT_CANCELLED = "client_cancelled"
TERMINATION_SERVER_TIMEOUT = "server_timeout"

_DONE_MARKER = "data: [DONE]\n\n"


def termination_frame(reason: str) -> str:
    """Return the SSE frame carrying the termination ``reason``."""
    return f'data: {{"termination_reason": "{reason}"}}\n\n'


def with_stream_termination(
    generator: AsyncGenerator[str, None],
    *,
    include_termination_reason: bool = False,
    timeout: float = 0.0,
    on_termination: Callable[[str], None] | None = None,
) -> AsyncGenerator[str, None]:
    """Report the termination reason of an SSE stream.

    Args:
        generator: The upstream SSE generator. It yields ``data: ...\n\n``
            strings and is expected to end with ``data: [DONE]\n\n``.
        include_termination_reason: If True, emit a termination event before
            the ``[DONE]`` marker. Defaults to False, leaving the wire format
            unchanged.
        timeout: Optional server-side deadline in seconds for the whole
            stream. ``0`` (the default) disables it.
        on_termination: Optional callback invoked exactly once with the
            termination reason (``stop``, ``client_cancelled`` or
            ``server_timeout``).
    """
    return _termination_stream(
        generator,
        include_termination_reason=include_termination_reason,
        timeout=timeout,
        on_termination=on_termination,
    )


async def _termination_stream(
    generator: AsyncGenerator[str, None],
    *,
    include_termination_reason: bool,
    timeout: float,
    on_termination: Callable[[str], None] | None,
) -> AsyncGenerator[str, None]:
    deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
    next_task: asyncio.Task[str] = asyncio.create_task(anext(generator))
    terminated = False

    def _finish(reason: str) -> None:
        nonlocal terminated
        if terminated:
            return
        terminated = True
        logger.info("Stream terminated with reason: %s", reason)
        if on_termination is not None:
            on_termination(reason)

    try:
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                done, _ = await asyncio.wait({next_task}, timeout=remaining)
                if not done:
                    raise asyncio.TimeoutError()
            else:
                done, _ = await asyncio.wait({next_task})
            chunk = next_task.result()
            if include_termination_reason and chunk.strip() == "data: [DONE]":
                yield termination_frame(TERMINATION_STOP)
            yield chunk
            if chunk.strip() == "data: [DONE]":
                # Terminal marker received; nothing more to emit.
                _finish(TERMINATION_STOP)
                return
            next_task = asyncio.create_task(anext(generator))
    except StopAsyncIteration:
        # Upstream exhausted without yielding another [DONE] (it already did).
        _finish(TERMINATION_STOP)
    except asyncio.TimeoutError:
        _finish(TERMINATION_SERVER_TIMEOUT)
        # Close the upstream generator so the engine aborts the request and
        # releases its resources before we emit the terminal frames.
        next_task.cancel()
        with contextlib.suppress(BaseException):
            await next_task
        with contextlib.suppress(BaseException):
            await generator.aclose()
        if include_termination_reason:
            yield termination_frame(TERMINATION_SERVER_TIMEOUT)
        yield _DONE_MARKER
    except (GeneratorExit, asyncio.CancelledError):
        # Client disconnected. Do not yield anything further; just make sure
        # the upstream is closed so the engine aborts and releases resources.
        _finish(TERMINATION_CLIENT_CANCELLED)
        raise
    finally:
        next_task.cancel()
        with contextlib.suppress(BaseException):
            await next_task
        with contextlib.suppress(BaseException):
            await generator.aclose()


class TerminationStreamingResponse(StreamingResponse):
    """A :class:`StreamingResponse` that closes its body iterator when the
    client disconnects.

    Starlette's default :class:`StreamingResponse` cancels the streaming task
    on disconnect but does not close the body iterator, so an async generator
    wrapped by :func:`with_stream_termination` would only be finalized later by
    garbage collection. This subclass closes the iterator promptly, allowing
    the engine request to be aborted and its resources released.
    """

    async def listen_for_disconnect(self, receive) -> None:  # type: ignore[override]
        await super().listen_for_disconnect(receive)
        with contextlib.suppress(BaseException):
            await self.body_iterator.aclose()
