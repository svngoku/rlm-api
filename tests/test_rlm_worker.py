from __future__ import annotations

import asyncio

from memory import Memory
from rlm_worker import build_rlm_query, format_memory_pack, recall_memories


class FakeMemory:
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


def test_parent_only_recalls_and_builds_prompt() -> None:
    memory = FakeMemory()
    recalled = asyncio.run(
        recall_memories(
            tenant_id="tenant-a",
            subject_id="subject-a",
            namespace="default",
            task="question",
            memory=memory,
        )
    )
    assert recalled[0].id == "memory-1"
    prompt = build_rlm_query("question", recalled)
    assert "remembered fact" in prompt
    assert "question" in prompt


def test_empty_memory_pack_is_explicit() -> None:
    assert "No relevant durable memory" in format_memory_pack([])
