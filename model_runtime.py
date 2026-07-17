"""DSPy model adapters; model identifiers remain entirely deployment-configured."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import multiprocessing
import os
import signal
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Protocol, cast

import dspy

from memory import JSONValue


@dataclass(frozen=True)
class RLMSubprocessInput:
    root_model_id: str
    sub_model_id: str
    context: str
    query: str
    max_iters: int
    max_llm_calls: int
    max_output_chars: int


@dataclass(frozen=True)
class RLMSubprocessResult:
    answer: str
    evidence: list[str]
    trajectory: JSONValue


class RemoteRLMError(RuntimeError):
    def __init__(self, exception_type: str, *, retryable: bool) -> None:
        super().__init__(exception_type)
        self.exception_type = exception_type
        self.retryable = retryable


class KillableProcess(Protocol):
    pid: int | None

    def is_alive(self) -> bool: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...


class DSPyEmbeddingProvider:
    """Async adapter around DSPy's provider-aware Embedder boundary."""

    def __init__(self, model_id: str) -> None:
        self._embedder = dspy.Embedder(model_id)

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        # DSPy/provider return types vary (lists and numpy arrays are both common).
        result: Any = await self._embedder.acall(list(texts))
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
    iteration_keyword = rlm_iteration_keyword(inspect.signature(dspy.RLM.__init__).parameters)
    options: dict[str, object] = {
        iteration_keyword: max_iters,
        "max_llm_calls": max_llm_calls,
        "max_output_chars": max_output_chars,
        "sub_lm": sub_lm,
    }
    program = dspy.RLM("context, query -> answer, evidence: list[str]", **options)
    return root_lm, program


def rlm_iteration_keyword(parameter_names: Collection[str]) -> str:
    if "max_iterations" in parameter_names:
        return "max_iterations"
    if "max_iters" in parameter_names:
        return "max_iters"
    raise RuntimeError("DSPy RLM exposes no supported iteration limit parameter")


async def execute_rlm_subprocess(inputs: RLMSubprocessInput) -> RLMSubprocessResult:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_rlm_subprocess_entrypoint,
        args=(inputs, sender),
        daemon=True,
    )
    process.start()
    sender.close()
    try:
        while True:
            if receiver.poll():
                payload: object = receiver.recv()
                return decode_subprocess_payload(payload)
            if not process.is_alive():
                if receiver.poll():
                    payload = receiver.recv()
                    return decode_subprocess_payload(payload)
                raise RemoteRLMError("ChildProcessExit", retryable=True)
            await asyncio.sleep(0.05)
    finally:
        receiver.close()
        await terminate_and_join(process)


async def terminate_and_join(process: KillableProcess) -> None:
    if process.is_alive():
        _signal_process_tree(process, signal.SIGTERM)
    await asyncio.to_thread(process.join, 5.0)
    if process.is_alive():
        _signal_process_tree(process, signal.SIGKILL)
        await asyncio.to_thread(process.join, 5.0)


def decode_subprocess_payload(payload: object) -> RLMSubprocessResult:
    if not isinstance(payload, dict):
        raise RemoteRLMError("InvalidChildPayload", retryable=True)
    if payload.get("ok") is False:
        exception_type = payload.get("exception_type")
        retryable = payload.get("retryable")
        if not isinstance(exception_type, str) or not isinstance(retryable, bool):
            raise RemoteRLMError("InvalidChildPayload", retryable=True)
        raise RemoteRLMError(exception_type, retryable=retryable)
    answer = payload.get("answer")
    evidence = payload.get("evidence")
    if (
        not isinstance(answer, str)
        or not isinstance(evidence, list)
        or not all(isinstance(item, str) for item in evidence)
    ):
        raise RemoteRLMError("InvalidChildResult", retryable=False)
    return RLMSubprocessResult(
        answer=answer,
        evidence=evidence,
        trajectory=_json_value(payload.get("trajectory")),
    )


def _rlm_subprocess_entrypoint(inputs: RLMSubprocessInput, sender: Connection) -> None:
    if hasattr(os, "setsid"):
        with contextlib.suppress(OSError):
            os.setsid()
    try:
        payload = asyncio.run(_execute_rlm_in_child(inputs))
    except BaseException as error:
        payload = {
            "ok": False,
            "exception_type": type(error).__name__,
            "retryable": child_exception_is_retryable(error),
        }
    try:
        sender.send(payload)
    finally:
        sender.close()


async def _execute_rlm_in_child(inputs: RLMSubprocessInput) -> dict[str, object]:
    root_lm, program = build_rlm(
        root_model_id=inputs.root_model_id,
        sub_model_id=inputs.sub_model_id,
        max_iters=inputs.max_iters,
        max_llm_calls=inputs.max_llm_calls,
        max_output_chars=inputs.max_output_chars,
    )
    with dspy.context(lm=root_lm):
        prediction = await program.aforward(context=inputs.context, query=inputs.query)
    if not isinstance(prediction.answer, str):
        raise ValueError("invalid answer type")
    if not isinstance(prediction.evidence, list) or not all(
        isinstance(item, str) for item in prediction.evidence
    ):
        raise ValueError("invalid evidence type")
    return {
        "ok": True,
        "answer": prediction.answer,
        "evidence": prediction.evidence,
        "trajectory": _json_value(getattr(prediction, "trajectory", None)),
    }


NONRETRYABLE_CHILD_EXCEPTIONS = frozenset(
    {
        "authenticationerror",
        "badrequesterror",
        "contextpolicyerror",
        "contextwindowexceedederror",
        "contentpolicyviolationerror",
        "keyerror",
        "permissiondeniederror",
        "permissionerror",
        "typeerror",
        "unprocessableentityerror",
        "valueerror",
    }
)


def child_exception_is_retryable(error: BaseException) -> bool:
    return not any(
        error_type.__name__.lower() in NONRETRYABLE_CHILD_EXCEPTIONS
        for error_type in type(error).mro()
    )


def _signal_process_tree(process: KillableProcess, signum: signal.Signals) -> None:
    if os.name == "posix" and process.pid is not None:
        try:
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            pass
    if signum is signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)
