# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ConcurrencyLimitMiddleware.

These tests drive the middleware directly through the ASGI interface (scope /
receive / send) so they don't require a running server or extra HTTP client
dependencies.
"""

import asyncio

import pytest

from vllm.entrypoints.openai.server_utils import (
    CONCURRENCY_LIMIT_EXCLUDED_ENDPOINTS,
    ConcurrencyLimitMiddleware,
)


def _http_scope(path: str, root_path: str = "") -> dict:
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "root_path": root_path,
        "headers": [],
    }


async def _drain(app, scope) -> int:
    """Run an ASGI app once and return the response status code."""
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    for message in sent:
        if message["type"] == "http.response.start":
            return message["status"]
    raise AssertionError("no http.response.start message was sent")


class _BlockingApp:
    """Downstream ASGI app whose requests block until released.

    Lets us hold a controllable number of requests "in flight" to exercise the
    concurrency limit deterministically.
    """

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0

    async def __call__(self, scope, receive, send):
        self.active += 1
        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})


@pytest.mark.asyncio
async def test_returns_429_when_limit_reached():
    downstream = _BlockingApp()
    mw = ConcurrencyLimitMiddleware(downstream, max_concurrency=1)

    # First request enters downstream and blocks, occupying the single slot.
    first = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)

    # Second request should fail fast with 429 without reaching downstream.
    status = await _drain(mw, _http_scope("/v1/completions"))
    assert status == 429
    assert downstream.active == 1

    # Release the first request and confirm it completes with 200.
    downstream.release.set()
    assert await asyncio.wait_for(first, timeout=1.0) == 200


@pytest.mark.asyncio
async def test_counter_is_released_after_request():
    downstream = _BlockingApp()
    mw = ConcurrencyLimitMiddleware(downstream, max_concurrency=1)

    # Let requests through immediately for this test.
    downstream.release.set()

    # Run several sequential requests; the slot must be freed each time.
    for _ in range(3):
        assert await _drain(mw, _http_scope("/v1/completions")) == 200
    assert mw._active == 0


@pytest.mark.asyncio
async def test_excluded_paths_are_never_throttled():
    downstream = _BlockingApp()
    mw = ConcurrencyLimitMiddleware(downstream, max_concurrency=1)

    # Occupy the only slot with a blocking /v1 request.
    first = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)

    # Health endpoints are exempt: allow them through even though the limit is
    # reached. Use a non-blocking app so they return immediately.
    downstream.release.set()
    for path in CONCURRENCY_LIMIT_EXCLUDED_ENDPOINTS:
        assert await _drain(mw, _http_scope(path)) == 200

    assert await asyncio.wait_for(first, timeout=1.0) == 200


@pytest.mark.asyncio
async def test_excluded_paths_respect_root_path():
    downstream = _BlockingApp()
    downstream.release.set()
    mw = ConcurrencyLimitMiddleware(downstream, max_concurrency=1)

    # /health mounted under a root_path prefix must still be recognized.
    scope = _http_scope("/prefix/health", root_path="/prefix")
    assert await _drain(mw, scope) == 200


@pytest.mark.asyncio
async def test_custom_excluded_paths_override_default():
    downstream = _BlockingApp()
    mw = ConcurrencyLimitMiddleware(
        downstream, max_concurrency=1, excluded_paths=("/ping",)
    )

    # Occupy the only slot.
    first = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)

    # /health is no longer exempt (only /ping is), so it should be throttled.
    assert await _drain(mw, _http_scope("/health")) == 429

    downstream.release.set()
    assert await asyncio.wait_for(first, timeout=1.0) == 200
