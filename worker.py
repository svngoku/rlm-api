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

from dotenv import load_dotenv

from config import Settings
from logging_config import configure_logging
from logging_config import context as log_context
from memory import JSONValue, Memory, MemoryKind, MemoryStore
from model_runtime import (
    DSPyEmbeddingProvider,
    RemoteRLMError,
    RLMSubprocessInput,
    RLMSubprocessResult,
    execute_rlm_subprocess,
)
from rlm_worker import build_rlm_query, recall_memories
from run_store import (
    ClaimedRun,
    RunStatus,
    RunStore,
    SummaryOutboxItem,
)

logger = logging.getLogger("rlm.worker")


class PollFailureThresholdExceeded(RuntimeError):
    pass


async def process_claim(
    run: ClaimedRun,
    *,
    store: RunStore,
    memory: MemoryStore,
    stale_seconds: int,
) -> None:
    started = time.monotonic()
    lease_lost = asyncio.Event()
    execution_task = asyncio.create_task(_execute_claim(run, memory))
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            run,
            store,
            max(1.0, min(30.0, stale_seconds / 3)),
            execution_task=execution_task,
            lease_lost=lease_lost,
            stale_seconds=stale_seconds,
        )
    )
    try:
        prediction, recalled = await execution_task
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
            "root_model": _model_id(run.model_config, "root_model"),
            "sub_model": _model_id(run.model_config, "sub_model"),
        }
        trajectory = (
            _json_value(getattr(prediction, "trajectory", None)) if run.include_trajectory else None
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
            summary_content=prediction.answer[:12_000],
            summary_metadata={
                "run_id": run.id,
                "source_memory_ids": [item.id for item in recalled],
            },
        )
        await _stop_task(heartbeat_task)
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
        await _record_failure(run, store, "run_timeout", "TimeoutError", retryable=True)
    except asyncio.CancelledError:
        if lease_lost.is_set():
            logger.warning(
                "run_execution_cancelled_after_lease_loss",
                extra={"context": log_context(run_id=run.id, tenant_id=run.tenant_id)},
            )
            return
        raise
    except Exception as exc:
        logger.error(
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
        await _record_failure(
            run,
            store,
            "execution_failed",
            type(exc).__name__,
            retryable=is_retryable_failure(exc),
        )
    finally:
        await _stop_task(heartbeat_task)
        await _stop_task(execution_task)


async def _execute_claim(
    run: ClaimedRun, memory: MemoryStore
) -> tuple[RLMSubprocessResult, list[Memory]]:
    root_model_id = _model_id(run.model_config, "root_model")
    sub_model_id = _model_id(run.model_config, "sub_model")
    async with asyncio.timeout(run.limits["timeout_s"]):
        recalled = await recall_memories(
            tenant_id=run.tenant_id,
            subject_id=run.subject_id,
            namespace=run.namespace,
            task=run.task,
            memory=memory,
        )
        prediction = await execute_rlm_subprocess(
            RLMSubprocessInput(
                root_model_id=root_model_id,
                sub_model_id=sub_model_id,
                context=run.context,
                query=build_rlm_query(run.task, recalled),
                max_iters=run.limits["max_iters"],
                max_llm_calls=run.limits["max_llm_calls"],
                max_output_chars=run.limits["max_output_chars"],
            )
        )
    return prediction, recalled


async def _record_failure(
    run: ClaimedRun,
    store: RunStore,
    public_error: str,
    exception_type: str,
    *,
    retryable: bool,
) -> None:
    next_status = await store.fail(
        run,
        public_error=public_error,
        event_metadata={"exception_type": exception_type},
        retryable=retryable,
    )
    logger.warning(
        "run_retry_scheduled" if next_status is RunStatus.QUEUED else "run_failed",
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
    run: ClaimedRun,
    store: RunStore,
    interval_seconds: float,
    *,
    execution_task: asyncio.Task[object],
    lease_lost: asyncio.Event,
    stale_seconds: int,
) -> None:
    last_confirmed = time.monotonic()
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            owned = await store.heartbeat(run)
        except Exception as error:
            logger.error(
                "run_heartbeat_failed",
                extra={
                    "context": log_context(
                        run_id=run.id,
                        tenant_id=run.tenant_id,
                        exception_type=type(error).__name__,
                    )
                },
            )
            if time.monotonic() - last_confirmed < stale_seconds * 0.9:
                continue
            owned = False
        else:
            if owned:
                last_confirmed = time.monotonic()
                continue
        if not owned:
            lease_lost.set()
            logger.warning(
                "run_heartbeat_lease_lost",
                extra={"context": log_context(run_id=run.id, tenant_id=run.tenant_id)},
            )
            try:
                await store.add_event(
                    run.id,
                    "worker_lease_lost",
                    {"attempt": run.attempts, "worker_id": run.worker_id},
                )
            except Exception as error:
                logger.error(
                    "run_lease_loss_event_failed",
                    extra={
                        "context": log_context(
                            run_id=run.id,
                            tenant_id=run.tenant_id,
                            exception_type=type(error).__name__,
                        )
                    },
                )
            execution_task.cancel()
            return


async def _stop_task(task: asyncio.Task[object]) -> None:
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


NONRETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "authenticationerror",
        "badrequesterror",
        "contextpolicyerror",
        "contextwindowexceedederror",
        "contentpolicyviolationerror",
        "permissiondeniederror",
        "unprocessableentityerror",
    }
)


def is_retryable_failure(error: BaseException) -> bool:
    if isinstance(error, RemoteRLMError):
        return error.retryable
    if isinstance(error, (KeyError, PermissionError, TypeError, ValueError)):
        return False
    return not any(
        error_type.__name__.lower() in NONRETRYABLE_EXCEPTION_NAMES
        for error_type in type(error).mro()
    )


async def run_worker_loop(
    settings: Settings,
    *,
    store: RunStore,
    memory: MemoryStore,
    stop_event: asyncio.Event,
    worker_id: str | None = None,
) -> None:
    identity = worker_id or (f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}")
    last_recovery = 0.0
    consecutive_failures = 0
    logger.info("worker_started", extra={"context": {"worker_id": identity}})
    while not stop_event.is_set():
        try:
            cycle_result = await _poll_cycle_or_stop(
                settings,
                store=store,
                memory=memory,
                worker_id=identity,
                last_recovery=last_recovery,
                stop_event=stop_event,
            )
            if cycle_result is None:
                break
            last_recovery, processed = cycle_result
            consecutive_failures = 0
        except asyncio.CancelledError:
            raise
        except Exception:
            processed = 0
            consecutive_failures = next_poll_failure_count(
                consecutive_failures,
                max_failures=settings.worker_max_poll_failures,
            )
            logger.error(
                "worker_poll_failed",
                extra={
                    "context": {
                        "worker_id": identity,
                        "consecutive_failures": consecutive_failures,
                    }
                },
            )
        if processed == 0:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=settings.worker_poll_seconds)
    logger.info("worker_stopped", extra={"context": {"worker_id": identity}})


async def _poll_cycle_or_stop(
    settings: Settings,
    *,
    store: RunStore,
    memory: MemoryStore,
    worker_id: str,
    last_recovery: float,
    stop_event: asyncio.Event,
) -> tuple[float, int] | None:
    cycle_task = asyncio.create_task(
        _poll_cycle(
            settings,
            store=store,
            memory=memory,
            worker_id=worker_id,
            last_recovery=last_recovery,
        )
    )
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, _ = await asyncio.wait({cycle_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done and stop_task.result():
            cycle_task.cancel()
            await asyncio.gather(cycle_task, return_exceptions=True)
            return None
        return await cycle_task
    finally:
        for task in (cycle_task, stop_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(cycle_task, stop_task, return_exceptions=True)


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
        retried, failed = await store.recover_stale(stale_seconds=settings.worker_stale_seconds)
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
        summary = await store.claim_summary(
            worker_id=worker_id,
            stale_seconds=settings.worker_stale_seconds,
        )
        if summary is None:
            break
        processed += 1
        await _deliver_summary(summary, store=store, memory=memory)

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


async def _deliver_summary(
    item: SummaryOutboxItem, *, store: RunStore, memory: MemoryStore
) -> None:
    try:
        await memory.write(
            tenant_id=item.tenant_id,
            subject_id=item.subject_id,
            namespace=item.namespace,
            kind=MemoryKind.RUN_SUMMARY,
            content=item.content,
            metadata={key: _json_value(value) for key, value in item.metadata.items()},
            importance=0.55,
            source_run_id=item.run_id,
        )
        completed = await store.complete_summary(item)
        logger.info(
            "run_summary_delivered" if completed else "run_summary_completion_lease_lost",
            extra={
                "context": log_context(
                    run_id=item.run_id,
                    tenant_id=item.tenant_id,
                    attempt=item.attempts,
                )
            },
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        retried = await store.retry_summary(item, public_error="summary_delivery_failed")
        logger.warning(
            "run_summary_retry_scheduled",
            extra={
                "context": log_context(
                    run_id=item.run_id,
                    tenant_id=item.tenant_id,
                    attempt=item.attempts,
                    exception_type=type(error).__name__,
                    lease_retained=retried,
                )
            },
        )


def next_poll_failure_count(current: int, *, max_failures: int) -> int:
    updated = current + 1
    if updated >= max_failures:
        raise PollFailureThresholdExceeded("worker poll failure threshold reached")
    return updated


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
        await run_worker_loop(settings, store=store, memory=memory, stop_event=stop_event)
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
