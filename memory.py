"""Neon-backed hybrid memory store (semantic + full-text)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import asyncpg
import dspy


@dataclass(frozen=True)
class Memory:
    id: str
    kind: str
    content: str
    metadata: dict[str, Any]
    score: float


class MemoryStore:
    def __init__(self, pool: asyncpg.Pool, embedding_lm: dspy.LM):
        self.pool = pool
        self.embedding_lm = embedding_lm

    @classmethod
    async def create(cls, embedding_lm: dspy.LM) -> "MemoryStore":
        pool = await asyncpg.create_pool(
            dsn=os.environ["DATABASE_URL"],
            min_size=1,
            max_size=int(os.getenv("DB_POOL_SIZE", "10")),
            command_timeout=30,
            server_settings={"application_name": "rlm-memory-api"},
        )
        return cls(pool, embedding_lm)

    async def embed(self, text: str) -> list[float]:
        """Compute embedding via the configured embedding LM."""
        vectors = await self.embedding_lm.aembed([text])
        return vectors[0]

    async def search(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        namespace: str = "default",
        query: str,
        limit: int = 8,
    ) -> list[Memory]:
        embedding = await self.embed(query)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, kind::text, content, metadata, score
                FROM search_memories($1, $2, $3, $4, $5::vector, $6)
                """,
                tenant_id,
                subject_id,
                namespace,
                query,
                json.dumps(embedding),
                limit,
            )

            if rows:
                await conn.execute(
                    """
                    UPDATE memory_items
                    SET access_count     = access_count + 1,
                        last_accessed_at = now()
                    WHERE id = ANY($1::uuid[])
                    """,
                    [row["id"] for row in rows],
                )

        return [
            Memory(
                id=str(row["id"]),
                kind=row["kind"],
                content=row["content"],
                metadata=dict(row["metadata"]),
                score=float(row["score"]),
            )
            for row in rows
        ]

    async def write(
        self,
        *,
        tenant_id: str,
        subject_id: str,
        namespace: str = "default",
        kind: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float = 0.5,
        expires_at: str | None = None,
    ) -> str:
        embedding = await self.embed(content)

        async with self.pool.acquire() as conn:
            memory_id = await conn.fetchval(
                """
                INSERT INTO memory_items (
                    tenant_id, subject_id, namespace, kind, content,
                    metadata, embedding, importance, expires_at
                )
                VALUES (
                    $1, $2, $3, $4::memory_kind, $5,
                    $6::jsonb, $7::vector, $8, $9::timestamptz
                )
                RETURNING id
                """,
                tenant_id,
                subject_id,
                namespace,
                kind,
                content,
                json.dumps(metadata or {}),
                json.dumps(embedding),
                importance,
                expires_at,
            )

        return str(memory_id)

    async def close(self) -> None:
        await self.pool.close()
