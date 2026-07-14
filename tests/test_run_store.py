from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from run_store import (
    RunStatus,
    RunStore,
    decode_claimed_run,
    failure_status,
    retry_delay_seconds,
    summary_retry_delay_seconds,
)


def claimed_row() -> dict[str, object]:
    return {
        "id": "00000000-0000-0000-0000-000000000001",
        "tenant_id": "tenant-a",
        "subject_id": "subject-a",
        "namespace": "default",
        "task": "question",
        "context": "corpus",
        "limits": ('{"max_iters":2,"max_llm_calls":3,"max_output_chars":1000,"timeout_s":30}'),
        "model_config": (
            '{"root_model":"provider/root","sub_model":"provider/sub",'
            '"embedding_model":"provider/embed","embedding_dim":3}'
        ),
        "include_trajectory": True,
        "attempts": 1,
        "max_attempts": 3,
    }


class FakeGetPool:
    def __init__(self, include_trajectory: bool) -> None:
        now = datetime.now(UTC)
        self.row = {
            "id": "00000000-0000-0000-0000-000000000001",
            "status": "succeeded",
            "tenant_id": "tenant-a",
            "subject_id": "subject-a",
            "namespace": "default",
            "model_config": "{}",
            "limits": "{}",
            "result": '{"answer":"ok"}',
            "usage": "{}",
            "error": None,
            "attempts": 1,
            "max_attempts": 3,
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
            "memory_snapshot": "[]",
            "include_trajectory": include_trajectory,
            "trajectory": '[{"step":1}]',
        }

    async def fetchrow(self, query: str, run_id: str, tenant_id: str) -> dict[str, object]:
        return self.row


class AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object,
    ) -> None:
        return None


class FakeClaimConnection:
    def __init__(self) -> None:
        malformed = claimed_row()
        malformed["limits"] = "{}"
        valid = claimed_row()
        valid["id"] = "00000000-0000-0000-0000-000000000002"
        self.rows = [malformed, valid]
        self.quarantined = 0

    def transaction(self) -> AsyncContext:
        return AsyncContext(self)

    async def fetchrow(self, query: str, worker_id: str) -> dict[str, object] | None:
        return self.rows.pop(0) if self.rows else None

    async def fetchval(self, query: str, run_id: str) -> str:
        return run_id

    async def execute(self, query: str, *values: object) -> str:
        if "invalid_persisted_configuration" in query:
            self.quarantined += 1
        return "UPDATE 1"


class FakeClaimPool:
    def __init__(self) -> None:
        self.connection = FakeClaimConnection()

    def acquire(self) -> AsyncContext:
        return AsyncContext(self.connection)


def test_retry_backoff_is_exponential_and_bounded() -> None:
    assert [retry_delay_seconds(attempt) for attempt in range(1, 5)] == [
        1,
        2,
        4,
        8,
    ]
    assert retry_delay_seconds(50) == 300
    assert summary_retry_delay_seconds(1) == 1
    assert summary_retry_delay_seconds(50) == 900
    with pytest.raises(ValueError):
        retry_delay_seconds(0)


def test_failure_transition_retries_until_last_attempt() -> None:
    assert failure_status(1, 3) is RunStatus.QUEUED
    assert failure_status(2, 3) is RunStatus.QUEUED
    assert failure_status(3, 3) is RunStatus.FAILED
    assert failure_status(1, 3, retryable=False) is RunStatus.FAILED
    with pytest.raises(ValueError):
        failure_status(4, 3)


def test_claim_decoder_requires_complete_valid_configuration() -> None:
    claimed = decode_claimed_run(claimed_row(), worker_id="worker-1")
    assert claimed.limits["timeout_s"] == 30
    assert claimed.model_config["root_model"] == "provider/root"

    missing_limit = claimed_row()
    missing_limit["limits"] = '{"max_iters":2}'
    with pytest.raises(ValueError, match="missing"):
        decode_claimed_run(missing_limit, worker_id="worker-1")

    invalid_model = claimed_row()
    invalid_model["model_config"] = '{"root_model":"","embedding_dim":3}'
    with pytest.raises(ValueError):
        decode_claimed_run(invalid_model, worker_id="worker-1")


def test_get_only_exposes_requested_trajectory() -> None:
    without = RunStore(FakeGetPool(False))  # type: ignore[arg-type]
    result = asyncio.run(without.get(run_id="run-1", tenant_id="tenant-a"))
    assert result is not None
    assert "trajectory" not in result

    with_trajectory = RunStore(FakeGetPool(True))  # type: ignore[arg-type]
    result = asyncio.run(with_trajectory.get(run_id="run-1", tenant_id="tenant-a"))
    assert result is not None
    assert result["trajectory"] == [{"step": 1}]


def test_claim_quarantines_poison_and_continues_to_valid_row() -> None:
    pool = FakeClaimPool()
    store = RunStore(pool)  # type: ignore[arg-type]
    claimed = asyncio.run(store.claim(worker_id="worker-1", max_quarantined=3))
    assert claimed is not None
    assert claimed.id == "00000000-0000-0000-0000-000000000002"
    assert pool.connection.quarantined == 1
