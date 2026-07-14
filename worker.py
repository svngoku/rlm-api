"""Separate durable queue worker process."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import time
import uuid
from collections.abc import Mapping, Sequence

import dspy
from dotenv import load_dotenv

from config import Settings
from logging_config import configure_logging
from logging_config import context as log_context
from memory import JSONValue, MemoryStore
from model_runtime import DSPyEmbeddingProvider, build_rlm
from rlm_worker import execute_run
from run_store import ClaimedRun, RunStatus, RunStore

logger = logging.getLogger("rlm.worker")


async def process_claim(
    run: ClaimedRun,
    *,
    store: RunStore,
    memory: MemoryStore,
    stale_seconds: int,
) -> None:
    started = time.monotonic()
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(run, store, max(1.0, min(30.0, stale_seconds / 3)))
    )
    try:
        root_model_id = _model_id(run.model_config, "root_model")
        sub_model_id = _model_id(run.model_config, "sub_model")
        root_lm, rlm = build_rlm(
            root_model_id=root_model_id,
            sub_model_id=sub_model_id,
            max_iters=run.limits["max_iters"],
            max_llm_calls=run.limits["max_llm_calls"],
            max_output_chars=run.limits["max_output_chars"],
        )
        async with asyncio.timeout(run.limits["timeout_s"]):
            with dspy.context(lm=root_lm):
                prediction, recalled = await execute_run(
                    run_id=run.id,
                    tenant_id=run.tenant_id,
                    subject_id=run.subject_id,
                    namespace=run.namespace,
                    task=run.task,
                    corpus=run.context,
                    memory=memory,
                    rlm=rlm,
                )
        latency_ms = round((time.monotonic() - started) * 1000)
        if not isinstance(prediction.answer, str):
            raise ValueError("RLM prediction answer must be a string")
        if not isinstance(prediction.evidence, list) or not all(
            isinstance(item, str) for item in prediction.evidence
        ):
            raise ValueError("RLM prediction evidence must be a list of strings")
        usage: dict[str, object] = {
            "latency_ms": latency_ms,
            "attempt": run.attempts,
            "root_model": root_model_id,
            "sub_model": sub_model_id,
        }
        trajectory = (
            _json_value(getattr(prediction, "trajectory", None))
            if run.include_trajectory
            else None
        )
        persisted = await store.succeed(
            run,
            result={
                "answer": prediction.answer,
                "evidence": prediction.evidence,
            },
            usage=usage,
            recalled_memory_ids=[item.id for item in recalled],
            trajectory=trajectory,
        )
        logger.info(
            "run_succeeded" if persisted else "run_completion_lease_lost",
            extra={
                "context": log_context(
                    run_id=run.id,
                    tenant_id=run.tenant_id,
                    attempt=run.attempts,
                    latency_ms=latency_ms,
                )
            },
        )
    except TimeoutError:
        await _record_failure(run, store, "run_timeout", "TimeoutError")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception(
            "run_execution_failed",
            extra={
                "context": log_context(
                    run_id=run.id,
                    tenant_id=run.tenant_id,
                    attempt=run.attempts,
                    exception_type=type(exc).__name__,
                )
            },
        )
        await _record_failure(run, store, "execution_failed", type(exc).__name__)
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task


async def _record_failure(
    run: ClaimedRun, store: RunStore, public_error: str, exception_type: str
) -> None:
    next_status = await store.fail(
        run,
        public_error=public_error,
        event_metadata={"exception_type": exception_type},
    )
    logger.warning(
        "run_retry_scheduled"
        if next_status is RunStatus.QUEUED
        else "run_failed",
        extra={
            "context": log_context(
                run_id=run.id,
                tenant_id=run.tenant_id,
                attempt=run.attempts,
                next_status=next_status.value if next_status else "lease_lost",
            )
        },
    )


async def _heartbeat_loop(
    run: ClaimedRun, store: RunStore, interval_seconds: float
) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        if not await store.heartbeat(run):
            logger.warning(
                "run_heartbeat_lease_lost",
                extra={
                    "context": log_context(
                        run_id=run.id, tenant_id=run.tenant_id
                    )
                },
            )
            return


async def run_worker_loop(
    settings: Settings,
    *,
    store: RunStore,
    memory: MemoryStore,
    stop_event: asyncio.Event,
    worker_id: str | None = None,
) -> None:
    identity = worker_id or (
        f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    )
    last_recovery = 0.0
    logger.info("worker_started", extra={"context": {"worker_id": identity}})
    while not stop_event.is_set():
        try:
            last_recovery, processed = await _poll_cycle(
                settings,
                store=store,
                memory=memory,
                worker_id=identity,
                last_recovery=last_recovery,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            processed = 0
            logger.exception(
                "worker_poll_failed",
                extra={"context": {"worker_id": identity}},
            )
        if processed == 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    stop_event.wait(), timeout=settings.worker_poll_seconds
                )
    logger.info("worker_stopped", extra={"context": {"worker_id": identity}})


async def _poll_cycle(
    settings: Settings,
    *,
    store: RunStore,
    memory: MemoryStore,
    worker_id: str,
    last_recovery: float,
) -> tuple[float, int]:
    now = time.monotonic()
    if now - last_recovery >= max(10.0, settings.worker_stale_seconds / 2):
        retried, failed = await store.recover_stale(
            stale_seconds=settings.worker_stale_seconds
        )
        if retried or failed:
            logger.warning(
                "stale_runs_recovered",
                extra={
                    "context": {
                        "worker_id": worker_id,
                        "retried": retried,
                        "failed": failed,
                    }
                },
            )
        last_recovery = now

    processed = 0
    for _ in range(settings.worker_batch_size):
        run = await store.claim(worker_id=worker_id)
        if run is None:
            break
        processed += 1
        await process_claim(
            run,
            store=store,
            memory=memory,
            stale_seconds=settings.worker_stale_seconds,
        )
    return last_recovery, processed


async def main() -> None:
    load_dotenv()
    configure_logging()
    settings = Settings.from_env()
    store = await RunStore.create(
        settings.database_url,
        pool_size=settings.db_pool_size,
        application_name="rlm-worker-queue",
    )
    try:
        memory = await MemoryStore.create(
            database_url=settings.database_url,
            pool_size=settings.db_pool_size,
            embedding_provider=DSPyEmbeddingProvider(settings.embedding_model),
            embedding_dim=settings.embedding_dim,
            application_name="rlm-worker-memory",
        )
    except Exception:
        await store.close()
        raise
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop_event.set)
    try:
        await run_worker_loop(
            settings, store=store, memory=memory, stop_event=stop_event
        )
    finally:
        await memory.close()
        await store.close()


def _model_id(config: Mapping[str, str | int], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"persisted model configuration is missing {key}")
    return value


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


if __name__ == "__main__":
    asyncio.run(main())
