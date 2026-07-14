from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from run_store import RunStatus, RunStore

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is required for PostgreSQL integration tests",
)


def test_durable_queue_retry_stale_recovery_and_event_ordering() -> None:
    assert TEST_DATABASE_URL is not None
    asyncio.run(_exercise_durable_queue(TEST_DATABASE_URL))


async def _exercise_durable_queue(database_url: str) -> None:
    tenant_id = f"integration-{uuid.uuid4()}"
    store = await RunStore.create(
        database_url,
        pool_size=2,
        application_name="rlm-integration-test",
    )
    try:
        retry_run_id = await _enqueue(store, tenant_id, "retry")
        first_claim = await store.claim(worker_id="retry-worker-1")
        assert first_claim is not None
        assert first_claim.id == retry_run_id
        status = await store.fail(
            first_claim,
            public_error="transient_failure",
            event_metadata={"test": True},
            retryable=True,
        )
        assert status is RunStatus.QUEUED
        await store.pool.execute(
            "UPDATE rlm_runs SET available_at = now() WHERE id = $1::uuid",
            retry_run_id,
        )
        second_claim = await store.claim(worker_id="retry-worker-2")
        assert second_claim is not None
        assert second_claim.id == retry_run_id

        sequences = await store.pool.fetch(
            """
            SELECT sequence, event_type
            FROM rlm_events
            WHERE run_id = $1::uuid
            ORDER BY sequence
            """,
            retry_run_id,
        )
        sequence_values = [row["sequence"] for row in sequences]
        assert sequence_values == list(range(1, len(sequence_values) + 1))
        assert len(sequence_values) == len(set(sequence_values))
        assert [row["event_type"] for row in sequences] == [
            "queued",
            "started",
            "retry_scheduled",
            "started",
        ]

        stale_ids: list[str] = []
        for index in range(2):
            run_id = await _enqueue(store, tenant_id, f"stale-{index}")
            claimed = await store.claim(worker_id=f"stale-worker-{index}")
            assert claimed is not None
            assert claimed.id == run_id
            stale_ids.append(run_id)
            await store.pool.execute(
                """
                UPDATE rlm_runs
                SET worker_heartbeat_at = now() - interval '1 hour',
                    created_at = now() - make_interval(secs => $2)
                WHERE id = $1::uuid
                """,
                run_id,
                20 - index,
            )

        recovered, failed = await store.recover_stale(stale_seconds=60, batch_size=1)
        assert (recovered, failed) == (1, 0)
        statuses = await store.pool.fetch(
            """
            SELECT status, count(*) AS count
            FROM rlm_runs
            WHERE id = ANY($1::uuid[])
            GROUP BY status
            """,
            stale_ids,
        )
        counts = {row["status"]: row["count"] for row in statuses}
        assert counts == {"queued": 1, "running": 1}
    finally:
        await store.pool.execute("DELETE FROM rlm_runs WHERE tenant_id = $1", tenant_id)
        await store.close()


async def _enqueue(store: RunStore, tenant_id: str, suffix: str) -> str:
    return await store.enqueue(
        tenant_id=tenant_id,
        subject_id=f"subject-{suffix}",
        namespace="integration",
        task="test task",
        context="test context",
        limits={
            "max_iters": 2,
            "max_llm_calls": 3,
            "max_output_chars": 1_000,
            "timeout_s": 30,
        },
        model_config={
            "root_model": "provider/root",
            "sub_model": "provider/sub",
            "embedding_model": "provider/embed",
            "embedding_dim": 1536,
        },
        include_trajectory=False,
        max_attempts=3,
    )
