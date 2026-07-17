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


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ({"subject_id": "s" * 257, "q": "query"}, "invalid_subject_id"),
        (
            {"subject_id": "subject", "namespace": "n" * 129, "q": "query"},
            "invalid_namespace",
        ),
        ({"subject_id": "subject", "q": "q" * 20_001}, "invalid_query"),
        ({"subject_id": "subject", "q": "query", "limit": "21"}, "invalid_limit"),
    ],
)
def test_memory_search_bounds_return_specific_errors(values: dict[str, str], error: str) -> None:
    parsed, validation_error = app_module.validate_memory_search_params(values)
    assert parsed is None
    assert validation_error == error


def test_overlong_memory_search_never_reaches_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRequest:
        headers = {"Authorization": "Bearer test-key"}
        query_params = {"subject_id": "s" * 257, "q": "query"}

    class SearchStore:
        calls = 0

        async def search(self, **values: object) -> list[object]:
            self.calls += 1
            return []

    store = SearchStore()
    monkeypatch.setattr(app_module, "memory_store", store)
    response = asyncio.run(app_module.search_memory(FakeRequest()))
    assert response.status_code == 400
    assert json.loads(response.description) == {"error": "invalid_subject_id"}
    assert store.calls == 0
