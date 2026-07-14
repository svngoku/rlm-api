"""RLM execution worker — memory recall → DSPy RLM → memory write."""
from __future__ import annotations

from typing import Any

import dspy

from memory import Memory, MemoryStore


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
    memory: MemoryStore,
    rlm: dspy.RLM,
) -> tuple[Any, list[Memory]]:
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

    await memory.write(
        tenant_id=tenant_id,
        subject_id=subject_id,
        namespace=namespace,
        kind="run_summary",
        content=prediction.answer[:12_000],
        metadata={
            "run_id": run_id,
            "source_memory_ids": [m.id for m in recalled],
        },
        importance=0.55,
    )

    return prediction, recalled
