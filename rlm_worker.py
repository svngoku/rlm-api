"""Parent-process memory recall and prompt preparation."""

from __future__ import annotations

from typing import Protocol

from memory import Memory


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


def format_memory_pack(memories: list[Memory]) -> str:
    if not memories:
        return "No relevant durable memory was retrieved."

    entries = "\n\n".join(
        f"[{m.kind} | score={m.score:.3f} | id={m.id}]\n{m.content}" for m in memories
    )
    return f"""Relevant durable memory (potentially stale — prioritise corpus evidence):
{entries}

Do not expose internal memory IDs unless explicitly requested.
"""


def build_rlm_query(task: str, memories: list[Memory]) -> str:
    return f"""{task}

{format_memory_pack(memories)}

Return a concise answer and evidence grounded in the corpus.
"""


async def recall_memories(
    *,
    tenant_id: str,
    subject_id: str,
    namespace: str,
    task: str,
    memory: MemoryService,
) -> list[Memory]:
    return await memory.search(
        tenant_id=tenant_id,
        subject_id=subject_id,
        namespace=namespace,
        query=task,
        limit=8,
    )
