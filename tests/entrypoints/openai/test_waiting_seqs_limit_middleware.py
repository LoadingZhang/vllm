# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for WaitingSeqsLimitMiddleware.

These tests drive the middleware directly through the ASGI interface (scope /
receive / send) so they don't require a running server or extra HTTP client
dependencies.
"""

import asyncio

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families

from vllm.entrypoints.serve.instrumentator import metrics as metrics_module
from vllm.entrypoints.serve.utils.server_utils import (
    WAITING_LIMIT_EXCLUDED_ENDPOINTS,
    WaitingSeqsLimitMiddleware,
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
    waiting limit deterministically.
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


class _FailingApp:
    async def __call__(self, scope, receive, send):
        raise RuntimeError("downstream failure")


class _CountingApp:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1


@pytest.mark.asyncio
async def test_returns_429_when_running_and_waiting_limit_reached():
    downstream = _BlockingApp()
    mw = WaitingSeqsLimitMiddleware(downstream, max_num_seqs=1, max_waiting_seqs=1)

    # Two requests occupy the running and waiting capacity.
    first = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)
    second = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.sleep(0)

    # A third request should fail fast without reaching downstream.
    status = await _drain(mw, _http_scope("/v1/completions"))
    assert status == 429
    assert downstream.active == 2

    downstream.release.set()
    assert await asyncio.wait_for(first, timeout=1.0) == 200
    assert await asyncio.wait_for(second, timeout=1.0) == 200


@pytest.mark.asyncio
async def test_zero_waiting_seqs_allows_only_max_num_seqs():
    downstream = _BlockingApp()
    mw = WaitingSeqsLimitMiddleware(downstream, max_num_seqs=1, max_waiting_seqs=0)

    first = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)

    assert await _drain(mw, _http_scope("/v1/completions")) == 429

    downstream.release.set()
    assert await asyncio.wait_for(first, timeout=1.0) == 200


@pytest.mark.asyncio
async def test_counter_is_released_after_request():
    downstream = _BlockingApp()
    mw = WaitingSeqsLimitMiddleware(downstream, max_num_seqs=1, max_waiting_seqs=0)

    # Let requests through immediately for this test.
    downstream.release.set()

    # Run several sequential requests; the slot must be freed each time.
    for _ in range(3):
        assert await _drain(mw, _http_scope("/v1/completions")) == 200
    assert mw._unfinished == 0


@pytest.mark.asyncio
async def test_counter_is_released_after_exception():
    mw = WaitingSeqsLimitMiddleware(_FailingApp(), max_num_seqs=1, max_waiting_seqs=0)

    with pytest.raises(RuntimeError, match="downstream failure"):
        await _drain(mw, _http_scope("/v1/completions"))

    assert mw._unfinished == 0


@pytest.mark.asyncio
async def test_counter_is_released_after_cancellation():
    downstream = _BlockingApp()
    mw = WaitingSeqsLimitMiddleware(downstream, max_num_seqs=1, max_waiting_seqs=0)

    request = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request

    assert mw._unfinished == 0


@pytest.mark.asyncio
async def test_non_http_scope_is_not_counted():
    downstream = _CountingApp()
    mw = WaitingSeqsLimitMiddleware(downstream, max_num_seqs=1, max_waiting_seqs=0)
    scope = {"type": "websocket", "path": "/v1/realtime", "root_path": ""}

    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(message):
        pass

    await mw(scope, receive, send)

    assert downstream.calls == 1
    assert mw._unfinished == 0


@pytest.mark.asyncio
async def test_excluded_paths_are_never_throttled():
    downstream = _BlockingApp()
    mw = WaitingSeqsLimitMiddleware(downstream, max_num_seqs=1, max_waiting_seqs=0)

    # Occupy the only slot with a blocking /v1 request.
    first = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)

    # Health endpoints are exempt: allow them through even though the limit is
    # reached. Use a non-blocking app so they return immediately.
    downstream.release.set()
    for path in WAITING_LIMIT_EXCLUDED_ENDPOINTS:
        assert await _drain(mw, _http_scope(path)) == 200

    assert await asyncio.wait_for(first, timeout=1.0) == 200


@pytest.mark.asyncio
async def test_excluded_paths_respect_root_path():
    downstream = _BlockingApp()
    downstream.release.set()
    mw = WaitingSeqsLimitMiddleware(downstream, max_num_seqs=1, max_waiting_seqs=0)

    # /health mounted under a root_path prefix must still be recognized.
    scope = _http_scope("/prefix/health", root_path="/prefix")
    assert await _drain(mw, scope) == 200


@pytest.mark.asyncio
async def test_custom_excluded_paths_override_default():
    downstream = _BlockingApp()
    mw = WaitingSeqsLimitMiddleware(
        downstream,
        max_num_seqs=1,
        max_waiting_seqs=0,
        excluded_paths=("/ping",),
    )

    # Occupy the only slot.
    first = asyncio.create_task(_drain(mw, _http_scope("/v1/completions")))
    await asyncio.wait_for(downstream.entered.wait(), timeout=1.0)

    # /health is no longer exempt (only /ping is), so it should be throttled.
    assert await _drain(mw, _http_scope("/health")) == 429

    downstream.release.set()
    assert await asyncio.wait_for(first, timeout=1.0) == 200


@pytest.mark.asyncio
async def test_rejections_are_recorded_with_exact_http_status(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.setattr(metrics_module, "get_prometheus_registry", lambda: registry)

    app = FastAPI()
    app.add_middleware(
        WaitingSeqsLimitMiddleware,
        max_num_seqs=1,
        max_waiting_seqs=0,
    )
    metrics_module.attach_router(app)

    entered = asyncio.Event()
    release = asyncio.Event()

    @app.get("/blocked")
    async def blocked():
        entered.set()
        await release.wait()
        return {"status": "ok"}

    @app.get("/ok")
    async def ok():
        return {"status": "ok"}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.get("/blocked"))
        await asyncio.wait_for(entered.wait(), timeout=1.0)

        rejected = await client.get("/blocked")
        assert rejected.status_code == 429

        # Metrics scraping remains available while the request limit is full.
        metrics_while_full = await client.get("/metrics")
        assert metrics_while_full.status_code == 200

        release.set()
        assert (await asyncio.wait_for(first, timeout=1.0)).status_code == 200
        assert (await client.get("/ok")).status_code == 200
        metrics = await client.get("/metrics")

    samples = [
        sample
        for family in text_string_to_metric_families(metrics.text)
        if family.name == "http_requests"
        for sample in family.samples
        if sample.name == "http_requests_total"
    ]
    statuses = {sample.labels["status"] for sample in samples}
    assert "429" in statuses
    assert "200" in statuses
    assert "4xx" not in statuses
    assert "2xx" not in statuses
