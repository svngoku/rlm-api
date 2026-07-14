from __future__ import annotations

import asyncio

from memory import JSONValue, Memory, MemoryKind
from rlm_worker import execute_run, format_memory_pack


class FakePrediction:
    answer = "grounded answer"
    evidence = ["corpus line"]


class FakeRLM:
    def __init__(self) -> None:
        self.query = ""

    async def aforward(self, *, context: str, query: str) -> FakePrediction:
        assert context == "corpus"
        self.query = query
        return FakePrediction()


class FakeMemory:
    def __init__(self) -> None:
        self.write_kind: MemoryKind | None = None

    async def search(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[Memory]:
        return [Memory("memory-1", "fact", "remembered fact", {}, 0.9)]

    async def write(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        namespace: str,
        kind: MemoryKind,
        content: str,
        metadata: dict[str, JSONValue],
        importance: float,
        expires_at: str | None = None,
    ) -> str:
        self.write_kind = kind
        assert metadata["run_id"] == "run-1"
        return "summary-1"


def test_execution_recall_run_write_flow() -> None:
    memory = FakeMemory()
    rlm = FakeRLM()
    prediction, recalled = asyncio.run(
        execute_run(
            run_id="run-1",
            tenant_id="tenant-a",
            subject_id="subject-a",
            namespace="default",
            task="question",
            corpus="corpus",
            memory=memory,
            rlm=rlm,
        )
    )
    assert prediction.answer == "grounded answer"
    assert recalled[0].id == "memory-1"
    assert "remembered fact" in rlm.query
    assert memory.write_kind is MemoryKind.RUN_SUMMARY


def test_empty_memory_pack_is_explicit() -> None:
    assert "No relevant durable memory" in format_memory_pack([])
