from __future__ import annotations

import asyncio
import importlib
import json
import os

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://example.invalid/db")
os.environ.setdefault("RLM_ROOT_MODEL", "provider/root")
os.environ.setdefault("RLM_SUB_MODEL", "provider/sub")
os.environ.setdefault("EMBEDDING_MODEL", "provider/embed")
os.environ.setdefault("EMBEDDING_DIM", "3")
os.environ.setdefault("API_KEYS_JSON", '{"test-key":"tenant-a"}')

app_module = importlib.import_module("app")
_response = app_module._response


def test_response_is_robyn_json_response_with_required_headers() -> None:
    response = _response(
        202,
        {"id": "run-1", "status": "queued"},
        "request-1",
        {"Location": "/v1/rlm/runs/run-1"},
    )

    assert response.status_code == 202
    assert response.headers.get("content-type") == "application/json"
    assert response.headers.get("x-request-id") == "request-1"
    assert response.headers.get("location") == "/v1/rlm/runs/run-1"
    assert json.loads(response.description) == {
        "id": "run-1",
        "status": "queued",
    }


def test_health_fails_when_embedded_worker_has_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadyStore:
        async def ready(self) -> bool:
            return True

    class FakeRequest:
        headers: dict[str, str] = {}

    async def scenario() -> int:
        async def terminated_worker() -> None:
            raise RuntimeError("worker stopped")

        task = asyncio.create_task(terminated_worker())
        await asyncio.gather(task, return_exceptions=True)
        monkeypatch.setattr(app_module, "run_store", ReadyStore())
        monkeypatch.setattr(app_module, "embedded_task", task)
        response = await app_module.healthz(FakeRequest())
        return response.status_code

    assert asyncio.run(scenario()) == 503
