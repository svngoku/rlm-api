"""Neon-backed hybrid memory store (semantic + full-text)."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import asyncpg

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]


class MemoryKind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    EPISODE = "episode"
    RUN_SUMMARY = "run_summary"
    FEEDBACK = "feedback"


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class Memory:
    id: str
    kind: str
    content: str
    metadata: dict[str, JSONValue]
    score: float


class MemoryStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_provider: EmbeddingProvider,
        embedding_dim: int,
    ) -> None:
        self.pool = pool
        self.embedding_provider = embedding_provider
        self.embedding_dim = embedding_dim

    @classmethod
    async def create(
        cls,
        *,
        database_url: str,
        pool_size: int,
        embedding_provider: EmbeddingProvider,
        embedding_dim: int,
        application_name: str = "rlm-memory",
    ) -> MemoryStore:
        pool = await asyncpg.create_pool(
            dsn=database_url,
            min_size=1,
            max_size=pool_size,
            command_timeout=30,
            server_settings={"application_name": application_name},
        )
        store = cls(pool, embedding_provider, embedding_dim)
        try:
            await store.validate_database_dimension()
        except Exception:
            await pool.close()
            raise
        return store

    async def embed(self, text: str) -> list[float]:
        vectors = await self.embedding_provider.embed([text])
        if len(vectors) != 1:
            raise ValueError("embedding provider returned an unexpected vector count")
        vector = [float(value) for value in vectors[0]]
        if len(vector) != self.embedding_dim:
            raise ValueError(
                "embedding dimension mismatch: "
                f"expected {self.embedding_dim}, received {len(vector)}"
            )
        return vector

    async def validate_database_dimension(self) -> None:
        declared_type = await self.pool.fetchval(
            """
            SELECT format_type(attribute.atttypid, attribute.atttypmod)
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            WHERE relation.relname = 'memory_items'
              AND attribute.attname = 'embedding'
              AND NOT attribute.attisdropped
            """
        )
        match = re.fullmatch(r"vector\((\d+)\)", str(declared_type or ""))
        if match is None:
            raise RuntimeError("memory_items.embedding is missing or is not fixed vector")
        database_dim = int(match.group(1))
        if database_dim != self.embedding_dim:
            raise RuntimeError(
                "EMBEDDING_DIM does not match memory_items.embedding: "
                f"configured {self.embedding_dim}, database {database_dim}"
            )

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
                metadata=_json_metadata(row["metadata"]),
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
        kind: MemoryKind,
        content: str,
        metadata: dict[str, JSONValue] | None = None,
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
                kind.value,
                content,
                json.dumps(metadata or {}),
                json.dumps(embedding),
                importance,
                expires_at,
            )

        return str(memory_id)

    async def close(self) -> None:
        await self.pool.close()


def _json_metadata(value: object) -> dict[str, JSONValue]:
    decoded: object = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, dict):
        raise ValueError("memory metadata is not a JSON object")
    return {str(key): item for key, item in decoded.items()}
