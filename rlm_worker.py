"""RLM execution worker — memory recall → DSPy RLM → memory write."""
from __future__ import annotations

from typing import Protocol

from memory import JSONValue, Memory, MemoryKind


class PredictionLike(Protocol):
    answer: str
    evidence: list[str]


class RLMProgram(Protocol):
    async def aforward(self, *, context: str, query: str) -> PredictionLike: ...


class MemoryService(Protocol):
    async def search(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        namespace: str,
        query: str,
        limit: int,
    ) -> list[Memory]: ...

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
    ) -> str: ...


def format_memory_pack(memories: list[Memory]) -> str:
    if not memories:
        return "No relevant durable memory was retrieved."

    entries = "\n\n".join(
        f"[{m.kind} | score={m.score:.3f} | id={m.id}]\n{m.content}"
        for m in memories
    )
    return f"""Relevant durable memory (potentially stale — prioritise corpus evidence):
{entries}

Do not expose internal memory IDs unless explicitly requested.
"""


async def execute_run(
    *,
    run_id: str,
    tenant_id: str,
    subject_id: str,
    namespace: str,
    task: str,
    corpus: str,
    memory: MemoryService,
    rlm: RLMProgram,
) -> tuple[PredictionLike, list[Memory]]:
    recalled = await memory.search(
        tenant_id=tenant_id,
        subject_id=subject_id,
        namespace=namespace,
        query=task,
        limit=8,
    )

    prediction = await rlm.aforward(
        context=corpus,
        query=f"""{task}

{format_memory_pack(recalled)}

Return a concise answer and evidence grounded in the corpus.
""",
    )
    if not isinstance(prediction.answer, str):
        raise ValueError("RLM prediction answer must be a string")
    if not isinstance(prediction.evidence, list) or not all(
        isinstance(item, str) for item in prediction.evidence
    ):
        raise ValueError("RLM prediction evidence must be a list of strings")

    await memory.write(
        tenant_id=tenant_id,
        subject_id=subject_id,
        namespace=namespace,
        kind=MemoryKind.RUN_SUMMARY,
        content=prediction.answer[:12_000],
        metadata={
            "run_id": run_id,
            "source_memory_ids": [m.id for m in recalled],
        },
        importance=0.55,
    )

    return prediction, recalled
