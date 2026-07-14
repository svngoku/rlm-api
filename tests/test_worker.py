from __future__ import annotations

import asyncio

import pytest

import worker
from memory import Memory
from model_runtime import RLMSubprocessResult
from run_store import ClaimedRun, SummaryOutboxItem


class FakeStore:
    def __init__(self, *, succeed_result: bool) -> None:
        self.succeed_result = succeed_result
        self.succeed_calls = 0
        self.fail_calls = 0
        self.events: list[str] = []
        self.complete_summary_calls = 0
        self.retry_summary_calls = 0

    async def succeed(self, run: ClaimedRun, **values: object) -> bool:
        self.succeed_calls += 1
        return self.succeed_result

    async def fail(self, run: ClaimedRun, **values: object) -> None:
        self.fail_calls += 1

    async def heartbeat(self, run: ClaimedRun) -> bool:
        return True

    async def add_event(self, run_id: str, event_type: str, payload: dict[str, object]) -> None:
        self.events.append(event_type)

    async def complete_summary(self, item: SummaryOutboxItem) -> bool:
        self.complete_summary_calls += 1
        return True

    async def retry_summary(self, item: SummaryOutboxItem, *, public_error: str) -> bool:
        self.retry_summary_calls += 1
        return True


class FakeMemory:
    def __init__(self, *, fail_write: bool = False) -> None:
        self.fail_write = fail_write
        self.write_calls = 0

    async def write(self, **values: object) -> str:
        self.write_calls += 1
        if self.fail_write:
            raise RuntimeError("summary database unavailable")
        return "memory-1"


def claimed_run() -> ClaimedRun:
    return ClaimedRun(
        id="00000000-0000-0000-0000-000000000001",
        tenant_id="tenant-a",
        subject_id="subject-a",
        namespace="default",
        task="question",
        context="corpus",
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
            "embedding_dim": 3,
        },
        include_trajectory=False,
        attempts=1,
        max_attempts=3,
        worker_id="worker-1",
    )


async def fake_execute(
    run: ClaimedRun, memory: FakeMemory
) -> tuple[RLMSubprocessResult, list[Memory]]:
    return RLMSubprocessResult("answer", ["evidence"], None), [
        Memory("source-1", "fact", "content", {}, 0.8)
    ]


def summary_item() -> SummaryOutboxItem:
    return SummaryOutboxItem(
        run_id="00000000-0000-0000-0000-000000000001",
        tenant_id="tenant-a",
        subject_id="subject-a",
        namespace="default",
        content="summary",
        metadata={"run_id": "00000000-0000-0000-0000-000000000001"},
        attempts=1,
        worker_id="worker-1",
    )


def test_summary_is_not_written_when_success_lease_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "_execute_claim", fake_execute)
    store = FakeStore(succeed_result=False)
    memory = FakeMemory()

    asyncio.run(
        worker.process_claim(
            claimed_run(),
            store=store,  # type: ignore[arg-type]
            memory=memory,  # type: ignore[arg-type]
            stale_seconds=600,
        )
    )

    assert store.succeed_calls == 1
    assert store.fail_calls == 0
    assert memory.write_calls == 0


def test_success_only_enqueues_summary_without_direct_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "_execute_claim", fake_execute)
    store = FakeStore(succeed_result=True)
    memory = FakeMemory()

    asyncio.run(
        worker.process_claim(
            claimed_run(),
            store=store,  # type: ignore[arg-type]
            memory=memory,  # type: ignore[arg-type]
            stale_seconds=600,
        )
    )

    assert store.succeed_calls == 1
    assert memory.write_calls == 0
    assert store.fail_calls == 0


def test_outbox_summary_success_and_failure_are_durable() -> None:
    async def scenario() -> tuple[FakeStore, FakeStore]:
        success_store = FakeStore(succeed_result=True)
        await worker._deliver_summary(
            summary_item(),
            store=success_store,  # type: ignore[arg-type]
            memory=FakeMemory(),  # type: ignore[arg-type]
        )
        retry_store = FakeStore(succeed_result=True)
        await worker._deliver_summary(
            summary_item(),
            store=retry_store,  # type: ignore[arg-type]
            memory=FakeMemory(fail_write=True),  # type: ignore[arg-type]
        )
        return success_store, retry_store

    success_store, retry_store = asyncio.run(scenario())
    assert success_store.complete_summary_calls == 1
    assert success_store.retry_summary_calls == 0
    assert retry_store.complete_summary_calls == 0
    assert retry_store.retry_summary_calls == 1


def test_heartbeat_lease_loss_records_event_and_cancels_execution() -> None:
    async def scenario() -> tuple[bool, list[str]]:
        store = FakeStore(succeed_result=False)

        async def false_heartbeat(run: ClaimedRun) -> bool:
            return False

        store.heartbeat = false_heartbeat  # type: ignore[method-assign]
        execution = asyncio.create_task(asyncio.sleep(60))
        lease_lost = asyncio.Event()
        await worker._heartbeat_loop(
            claimed_run(),
            store,  # type: ignore[arg-type]
            0,
            execution_task=execution,
            lease_lost=lease_lost,
            stale_seconds=600,
        )
        await asyncio.gather(execution, return_exceptions=True)
        return lease_lost.is_set() and execution.cancelled(), store.events

    cancelled, events = asyncio.run(scenario())
    assert cancelled
    assert events == ["worker_lease_lost"]


def test_transient_heartbeat_exception_does_not_cancel_execution() -> None:
    async def scenario() -> bool:
        store = FakeStore(succeed_result=False)
        calls = 0
        execution = asyncio.create_task(asyncio.sleep(60))

        async def uncertain_then_lost(run: ClaimedRun) -> bool:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("transient database failure")
            assert not execution.cancelled()
            return False

        store.heartbeat = uncertain_then_lost  # type: ignore[method-assign]
        lease_lost = asyncio.Event()
        await worker._heartbeat_loop(
            claimed_run(),
            store,  # type: ignore[arg-type]
            0,
            execution_task=execution,
            lease_lost=lease_lost,
            stale_seconds=600,
        )
        await asyncio.gather(execution, return_exceptions=True)
        return calls == 2 and lease_lost.is_set() and execution.cancelled()

    assert asyncio.run(scenario())


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad local config"),
        TypeError("bad value"),
        KeyError("missing"),
        PermissionError("denied"),
    ],
)
def test_local_validation_failures_are_not_retryable(error: Exception) -> None:
    assert not worker.is_retryable_failure(error)


def test_provider_bad_request_name_is_not_retryable() -> None:
    bad_request_type = type("BadRequestError", (Exception,), {})
    assert not worker.is_retryable_failure(bad_request_type())
    assert worker.is_retryable_failure(RuntimeError("transient provider failure"))
    assert worker.is_retryable_failure(TimeoutError())


def test_poll_failure_threshold_is_bounded() -> None:
    assert worker.next_poll_failure_count(0, max_failures=3) == 1
    assert worker.next_poll_failure_count(1, max_failures=3) == 2
    with pytest.raises(worker.PollFailureThresholdExceeded):
        worker.next_poll_failure_count(2, max_failures=3)


def test_stop_event_cancels_active_poll_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancelled = asyncio.Event()

    async def hanging_cycle(*args: object, **kwargs: object) -> tuple[float, int]:
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()
        return 0.0, 0

    async def scenario() -> bool:
        monkeypatch.setattr(worker, "_poll_cycle", hanging_cycle)
        stop = asyncio.Event()
        task = asyncio.create_task(
            worker._poll_cycle_or_stop(
                object(),  # type: ignore[arg-type]
                store=object(),  # type: ignore[arg-type]
                memory=object(),  # type: ignore[arg-type]
                worker_id="worker-1",
                last_recovery=0.0,
                stop_event=stop,
            )
        )
        await asyncio.sleep(0)
        stop.set()
        result = await task
        return result is None and cancelled.is_set()

    assert asyncio.run(scenario())
