from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from model_runtime import (
    DSPyEmbeddingProvider,
    RemoteRLMError,
    build_rlm,
    decode_subprocess_payload,
    rlm_iteration_keyword,
    terminate_and_join,
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def acall(self, inputs: list[str]) -> Sequence[Sequence[float]]:
        self.inputs = inputs
        return [[1.0, 2.0, 3.0]]


class FakeProcess:
    def __init__(self) -> None:
        self.pid: int | None = None
        self.alive = True
        self.terminated = False
        self.joined = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.joined = True


def test_rlm_iteration_keyword_supports_current_and_future_names() -> None:
    assert rlm_iteration_keyword({"max_iterations"}) == "max_iterations"
    assert rlm_iteration_keyword({"max_iters"}) == "max_iters"


def test_build_rlm_uses_installed_dspy_constructor_without_provider_call() -> None:
    _, program = build_rlm(
        root_model_id="provider/operator-selected-root",
        sub_model_id="provider/operator-selected-sub",
        max_iters=7,
        max_llm_calls=9,
        max_output_chars=1_000,
    )
    assert program.max_iterations == 7
    assert program.max_llm_calls == 9


def test_embedding_adapter_uses_async_embedder_call() -> None:
    provider = DSPyEmbeddingProvider("provider/operator-selected-embedding")
    fake = FakeEmbedder()
    provider._embedder = fake  # type: ignore[assignment]
    vectors = asyncio.run(provider.embed(["input"]))
    assert fake.inputs == ["input"]
    assert vectors == [[1.0, 2.0, 3.0]]


def test_subprocess_payload_is_validated_without_raw_error_message() -> None:
    result = decode_subprocess_payload(
        {
            "ok": True,
            "answer": "answer",
            "evidence": ["source"],
            "trajectory": [{"step": 1}],
        }
    )
    assert result.answer == "answer"
    assert result.trajectory == [{"step": 1}]

    with pytest.raises(RemoteRLMError) as captured:
        decode_subprocess_payload(
            {
                "ok": False,
                "exception_type": "AuthenticationError",
                "retryable": False,
            }
        )
    assert captured.value.exception_type == "AuthenticationError"
    assert not hasattr(captured.value, "raw_message")


def test_process_cleanup_terminates_and_joins_child() -> None:
    process = FakeProcess()
    asyncio.run(terminate_and_join(process))
    assert process.terminated
    assert process.joined
