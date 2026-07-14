"""DSPy model adapters; model identifiers remain entirely deployment-configured."""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import dspy


class DSPyEmbeddingProvider:
    """Async adapter around DSPy's synchronous Embedder boundary."""

    def __init__(self, model_id: str) -> None:
        self._embedder = dspy.Embedder(model_id)

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        # DSPy/provider return types vary (lists and numpy arrays are both common).
        result: Any = await asyncio.to_thread(self._embedder, list(texts))
        return cast(Sequence[Sequence[float]], result)


def build_rlm(
    *,
    root_model_id: str,
    sub_model_id: str,
    max_iters: int,
    max_llm_calls: int,
    max_output_chars: int,
) -> tuple[dspy.LM, dspy.RLM]:
    root_lm = dspy.LM(root_model_id)
    sub_lm = dspy.LM(sub_model_id)
    program = dspy.RLM(
        "context, query -> answer, evidence: list[str]",
        max_iterations=max_iters,
        max_llm_calls=max_llm_calls,
        max_output_chars=max_output_chars,
        sub_lm=sub_lm,
    )
    return root_lm, program
