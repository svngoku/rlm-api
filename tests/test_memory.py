from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from memory import MemoryStore


class FakePool:
    def __init__(self, declared_type: str = "vector(3)") -> None:
        self.declared_type = declared_type

    async def fetchval(self, query: str) -> str:
        return self.declared_type


class FakeEmbeddingProvider:
    def __init__(self, vector: Sequence[float]) -> None:
        self.vector = vector

    async def embed(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]:
        return [self.vector for _ in texts]


def test_embedding_length_is_validated_before_database_use() -> None:
    store = MemoryStore(FakePool(), FakeEmbeddingProvider([1.0, 2.0]), 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected 3, received 2"):
        asyncio.run(store.embed("text"))


def test_database_vector_dimension_is_validated() -> None:
    store = MemoryStore(FakePool("vector(4)"), FakeEmbeddingProvider([1, 2, 3]), 3)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="configured 3, database 4"):
        asyncio.run(store.validate_database_dimension())
