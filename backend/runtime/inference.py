"""
Shared inference runtime — keeps the API event loop responsive under load.

Why this exists
---------------
The API runs as a single Uvicorn process with one asyncio event loop. Any
synchronous, CPU- or GPU-bound work (loading a model, a torch forward/backward
pass, a blocking call to a cloud API) that runs *inline on the event loop* will
freeze the entire server for every other connected user until it finishes.

That is exactly what happened: an ``async def`` endpoint calling synchronous
PyTorch code blocked the loop, so even ``GET /`` timed out.

This module provides two things every blocking handler should use:

1. ``run_inference(fn, *args, **kwargs)`` — runs ``fn`` in a worker thread
   (off the event loop) *and* serializes GPU/heavy-CPU work behind a global
   lock. The lock matters because the whole world shares one GPU: without it,
   N concurrent requests would push N models/activation sets onto the device at
   once and trigger CUDA OOM or thrash CPU caches. Serializing keeps each
   request fast and predictable instead of everyone failing together.

2. ``inference_lock()`` — a context manager for code paths that are already off
   the event loop (e.g. the job worker, or a sync ``def`` route that Starlette
   already offloads to its threadpool) but still need to serialize GPU access.

Design notes
------------
* The lock is a plain re-entrant lock (``threading.RLock``) so a function that
  already holds it can call another helper that also acquires it without
  deadlocking.
* ``run_inference`` uses Starlette's ``run_in_threadpool`` so it integrates
  with the server's anyio threadpool rather than spawning unbounded threads.
* Pure I/O-bound fan-out (e.g. calling several cloud verifiers in parallel)
  should NOT hold the GPU lock — those callers pass ``serialize=False``.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("mlsec.runtime")

T = TypeVar("T")

# Global lock serializing GPU / heavy-CPU inference across all requests.
# Re-entrant so nested helper calls on the same thread don't deadlock.
_INFERENCE_LOCK = threading.RLock()


@contextmanager
def inference_lock(*, serialize: bool = True) -> Iterator[None]:
    """Serialize a block of GPU/heavy-CPU work.

    Use from code that is already running off the event loop (job worker, or a
    sync ``def`` route). When ``serialize`` is False this is a no-op, which is
    handy for callers that fan work out across threads themselves.
    """
    if not serialize:
        yield
        return
    acquired = _INFERENCE_LOCK.acquire()
    try:
        yield
    finally:
        if acquired:
            _INFERENCE_LOCK.release()


def run_inference_sync(fn: Callable[..., T], *args: Any, serialize: bool = True, **kwargs: Any) -> T:
    """Run a blocking callable under the inference lock (synchronous).

    For callers that are already off the event loop and just want the
    serialization guarantee.
    """
    with inference_lock(serialize=serialize):
        return fn(*args, **kwargs)


async def run_inference(fn: Callable[..., T], *args: Any, serialize: bool = True, **kwargs: Any) -> T:
    """Run a blocking callable off the event loop, serialized on the GPU lock.

    This is the helper that ``async def`` route handlers must use for any
    PyTorch / model-loading / heavy work so the event loop stays free to serve
    other users. Acquiring the lock happens *inside* the worker thread, so the
    event loop never blocks waiting for the lock either.
    """

    def _runner() -> T:
        with inference_lock(serialize=serialize):
            return fn(*args, **kwargs)

    return await run_in_threadpool(_runner)


_QUEUE_SENTINEL = object()


async def iterate_in_threadpool(
    gen_factory: Callable[..., Any],
    *args: Any,
    serialize: bool = False,
    **kwargs: Any,
):
    """Drive a *synchronous* generator from a worker thread, yielding async.

    Streaming endpoints (e.g. SSE benchmarks) consume a sync generator that
    does heavy torch work between yields. If that loop runs on the event loop
    it freezes the server between every item. This runs the generator in a
    worker thread and hands each produced item back to the event loop through a
    thread-safe queue, so the loop stays responsive.

    ``serialize`` defaults to False because long-running streams shouldn't hold
    the global GPU lock for their entire duration — the per-step inference
    inside the generator can take the lock itself if needed.
    """
    import asyncio
    import queue as _queue

    loop = asyncio.get_running_loop()
    q: "_queue.Queue[Any]" = _queue.Queue(maxsize=8)

    def _producer():
        try:
            with inference_lock(serialize=serialize):
                for item in gen_factory(*args, **kwargs):
                    q.put(item)
        except Exception as exc:  # surface producer errors to the consumer
            q.put((_QUEUE_SENTINEL, exc))
            return
        q.put(_QUEUE_SENTINEL)

    fut = loop.run_in_executor(None, _producer)
    try:
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _QUEUE_SENTINEL:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] is _QUEUE_SENTINEL:
                raise item[1]
            yield item
    finally:
        await fut
