"""Durable PostgreSQL run queue and lifecycle persistence."""
from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import asyncpg


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ClaimedRun:
    id: str
    tenant_id: str
    subject_id: str
    namespace: str
    task: str
    context: str
    limits: dict[str, int]
    model_config: dict[str, str | int]
    include_trajectory: bool
    attempts: int
    max_attempts: int
    worker_id: str


def retry_delay_seconds(attempts: int, *, cap_seconds: int = 300) -> int:
    """Bounded exponential retry delay after a failed claimed attempt."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    return min(cap_seconds, 2 ** min(attempts - 1, 20))


def failure_status(attempts: int, max_attempts: int) -> RunStatus:
    if attempts < 1 or max_attempts < 1 or attempts > max_attempts:
        raise ValueError("attempt counters are invalid")
    return RunStatus.QUEUED if attempts < max_attempts else RunStatus.FAILED


class RunStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def create(
        cls, database_url: str, *, pool_size: int, application_name: str
    ) -> RunStore:
        pool = await asyncpg.create_pool(
            dsn=database_url,
            min_size=1,
            max_size=pool_size,
            command_timeout=30,
            server_settings={"application_name": application_name},
        )
        return cls(pool)

    async def close(self) -> None:
        await self.pool.close()

    async def ready(self) -> bool:
        try:
            return bool(await self.pool.fetchval("SELECT 1"))
        except Exception:
            return False

    async def enqueue(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        namespace: str,
        task: str,
        context: str,
        limits: Mapping[str, int],
        model_config: Mapping[str, str | int],
        include_trajectory: bool,
        max_attempts: int,
    ) -> str:
        run_id = str(uuid.uuid4())
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """
                    INSERT INTO rlm_runs (
                        id, tenant_id, subject_id, namespace, status, task,
                        context, corpus_ref, model_config, limits,
                        include_trajectory, max_attempts, available_at, updated_at
                    )
                    VALUES (
                        $1::uuid, $2, $3, $4, 'queued', $5, $6,
                        jsonb_build_object('storage', 'inline'), $7::jsonb,
                        $8::jsonb, $9, $10, now(), now()
                    )
                    """,
                run_id,
                tenant_id,
                subject_id,
                namespace,
                task,
                context,
                json.dumps(dict(model_config)),
                json.dumps(dict(limits)),
                include_trajectory,
                max_attempts,
            )
            await self._append_event(
                connection, run_id, "queued", {"attempt": 0}
            )
        return run_id

    async def get(self, *, run_id: str, tenant_id: str) -> dict[str, object] | None:
        try:
            row = await self.pool.fetchrow(
                """
                SELECT id, status, tenant_id, subject_id, namespace, model_config,
                       limits, result, usage, error, attempts, max_attempts,
                       created_at, updated_at, started_at, completed_at,
                       memory_snapshot
                FROM rlm_runs
                WHERE id = $1::uuid AND tenant_id = $2
                """,
                run_id,
                tenant_id,
            )
        except asyncpg.InvalidTextRepresentationError:
            return None
        if row is None:
            return None
        return {
            "id": str(row["id"]),
            "status": row["status"],
            "tenant_id": row["tenant_id"],
            "subject_id": row["subject_id"],
            "namespace": row["namespace"],
            "model_config": _json_object(row["model_config"]),
            "limits": _json_object(row["limits"]),
            "result": _json_object(row["result"]) if row["result"] else None,
            "usage": _json_object(row["usage"]),
            "error": row["error"],
            "attempts": row["attempts"],
            "max_attempts": row["max_attempts"],
            "created_at": _iso(row["created_at"]),
            "updated_at": _iso(row["updated_at"]),
            "started_at": _iso(row["started_at"]),
            "completed_at": _iso(row["completed_at"]),
            "recalled_memory_ids": _json_list(row["memory_snapshot"]),
        }

    async def claim(self, *, worker_id: str) -> ClaimedRun | None:
        async with self.pool.acquire() as connection, connection.transaction():
            row = await connection.fetchrow(
                """
                WITH candidate AS (
                    SELECT id
                    FROM rlm_runs
                    WHERE status = 'queued'
                      AND available_at <= now()
                      AND attempts < max_attempts
                    ORDER BY available_at, created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                UPDATE rlm_runs AS run
                SET status = 'running',
                    attempts = run.attempts + 1,
                    worker_id = $1,
                    worker_heartbeat_at = now(),
                    started_at = COALESCE(run.started_at, now()),
                    updated_at = now()
                FROM candidate
                WHERE run.id = candidate.id
                RETURNING run.id, run.tenant_id, run.subject_id, run.namespace,
                          run.task, run.context, run.limits, run.model_config,
                          run.include_trajectory, run.attempts, run.max_attempts
                """,
                worker_id,
            )
            if row is None:
                return None
            claimed = ClaimedRun(
                id=str(row["id"]),
                tenant_id=row["tenant_id"],
                subject_id=row["subject_id"],
                namespace=row["namespace"],
                task=row["task"],
                context=row["context"],
                limits=_json_int_object(row["limits"]),
                model_config=_json_model_object(row["model_config"]),
                include_trajectory=row["include_trajectory"],
                attempts=row["attempts"],
                max_attempts=row["max_attempts"],
                worker_id=worker_id,
            )
            await self._append_event(
                connection,
                claimed.id,
                "started",
                {"attempt": claimed.attempts, "worker_id": worker_id},
            )
        return claimed

    async def heartbeat(self, run: ClaimedRun) -> bool:
        result = await self.pool.execute(
            """
            UPDATE rlm_runs
            SET worker_heartbeat_at = now(), updated_at = now()
            WHERE id = $1::uuid AND status = 'running' AND worker_id = $2
            """,
            run.id,
            run.worker_id,
        )
        return result == "UPDATE 1"

    async def succeed(
        self,
        run: ClaimedRun,
        *,
        result: Mapping[str, object],
        usage: Mapping[str, object],
        recalled_memory_ids: list[str],
        trajectory: object | None,
    ) -> bool:
        async with self.pool.acquire() as connection, connection.transaction():
            status = await connection.execute(
                """
                    UPDATE rlm_runs
                    SET status = 'succeeded', result = $3::jsonb, usage = $4::jsonb,
                        memory_snapshot = $5::jsonb,
                        trajectory = CASE WHEN include_trajectory THEN $6::jsonb
                                          ELSE NULL END,
                        error = NULL, completed_at = now(), updated_at = now(),
                        worker_id = NULL, worker_heartbeat_at = NULL
                    WHERE id = $1::uuid AND status = 'running' AND worker_id = $2
                    """,
                run.id,
                run.worker_id,
                json.dumps(dict(result)),
                json.dumps(dict(usage)),
                json.dumps(recalled_memory_ids),
                json.dumps(trajectory),
            )
            if status != "UPDATE 1":
                return False
            await self._append_event(
                connection,
                run.id,
                "succeeded",
                {"attempt": run.attempts, "usage": dict(usage)},
            )
        return True

    async def fail(
        self,
        run: ClaimedRun,
        *,
        public_error: str,
        event_metadata: Mapping[str, object],
    ) -> RunStatus | None:
        next_status = failure_status(run.attempts, run.max_attempts)
        retry = next_status is RunStatus.QUEUED
        delay = retry_delay_seconds(run.attempts)
        async with self.pool.acquire() as connection, connection.transaction():
            status = await connection.execute(
                """
                    UPDATE rlm_runs
                    SET status = $3, error = $4,
                        available_at = CASE
                            WHEN $3 = 'queued'
                            THEN now() + make_interval(secs => $5)
                            ELSE available_at
                        END,
                        completed_at = CASE WHEN $3 = 'failed' THEN now() ELSE NULL END,
                        updated_at = now(), worker_id = NULL,
                        worker_heartbeat_at = NULL
                    WHERE id = $1::uuid AND status = 'running' AND worker_id = $2
                    """,
                run.id,
                run.worker_id,
                next_status.value,
                public_error,
                delay,
            )
            if status != "UPDATE 1":
                return None
            payload = {
                "attempt": run.attempts,
                "retrying": retry,
                "retry_delay_seconds": delay if retry else None,
                **event_metadata,
            }
            await self._append_event(
                connection,
                run.id,
                "retry_scheduled" if retry else "failed",
                payload,
            )
        return next_status

    async def recover_stale(self, *, stale_seconds: int) -> tuple[int, int]:
        async with self.pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch(
                """
                    SELECT id, attempts, max_attempts
                    FROM rlm_runs
                    WHERE status = 'running'
                      AND COALESCE(
                            worker_heartbeat_at, updated_at, started_at, created_at
                          ) < now() - make_interval(secs => $1)
                    FOR UPDATE SKIP LOCKED
                    """,
                stale_seconds,
            )
            retried = 0
            failed = 0
            for row in rows:
                next_status = failure_status(
                    row["attempts"], row["max_attempts"]
                )
                retry = next_status is RunStatus.QUEUED
                status = next_status.value
                delay = retry_delay_seconds(row["attempts"])
                await connection.execute(
                    """
                        UPDATE rlm_runs
                        SET status = $2, error = 'worker_heartbeat_expired',
                            available_at = CASE
                                WHEN $2 = 'queued'
                                THEN now() + make_interval(secs => $3)
                                ELSE available_at
                            END,
                            completed_at = CASE WHEN $2 = 'failed' THEN now() ELSE NULL END,
                            worker_id = NULL, worker_heartbeat_at = NULL,
                            updated_at = now()
                        WHERE id = $1
                        """,
                    row["id"],
                    status,
                    delay,
                )
                await self._append_event(
                    connection,
                    str(row["id"]),
                    "stale_recovered" if retry else "failed",
                    {"attempt": row["attempts"], "retrying": retry},
                )
                retried += int(retry)
                failed += int(not retry)
        return retried, failed

    async def add_event(
        self, run_id: str, event_type: str, payload: Mapping[str, object]
    ) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await self._append_event(connection, run_id, event_type, payload)

    async def _append_event(
        self,
        connection: asyncpg.Connection,
        run_id: str,
        event_type: str,
        payload: Mapping[str, object],
    ) -> None:
        await connection.fetchval(
            "SELECT id FROM rlm_runs WHERE id = $1::uuid FOR UPDATE", run_id
        )
        await connection.execute(
            """
            INSERT INTO rlm_events (run_id, sequence, event_type, payload)
            SELECT $1::uuid, COALESCE(MAX(sequence), 0) + 1, $2, $3::jsonb
            FROM rlm_events
            WHERE run_id = $1::uuid
            """,
            run_id,
            event_type,
            json.dumps(dict(payload)),
        )


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _decode_json(value: object) -> object:
    if isinstance(value, str):
        decoded: object = json.loads(value)
        return decoded
    return value


def _json_object(value: object) -> dict[str, object]:
    decoded = _decode_json(value)
    if not isinstance(decoded, Mapping):
        raise ValueError("database JSON value is not an object")
    return {str(key): item for key, item in decoded.items()}


def _json_int_object(value: object) -> dict[str, int]:
    decoded = _json_object(value)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in decoded.values()):
        raise ValueError("persisted limits must contain integers")
    return {key: item for key, item in decoded.items() if isinstance(item, int)}


def _json_model_object(value: object) -> dict[str, str | int]:
    decoded = _json_object(value)
    if any(
        isinstance(item, bool) or not isinstance(item, (str, int))
        for item in decoded.values()
    ):
        raise ValueError("persisted model configuration has invalid values")
    return {
        key: item
        for key, item in decoded.items()
        if isinstance(item, (str, int)) and not isinstance(item, bool)
    }


def _json_list(value: object) -> list[object]:
    decoded = _decode_json(value)
    if not isinstance(decoded, list):
        raise ValueError("database JSON value is not an array")
    return decoded
