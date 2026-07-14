"""Robyn API — DSPy RLM runs + Neon memory layer."""
from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from time import time

import dspy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from robyn import Robyn, Request

from auth import require_auth
from memory import MemoryStore
from rlm_worker import execute_run

load_dotenv()

app = Robyn(__file__)

# --------------------------------------------------------------------------- #
# DSPy model configuration
# --------------------------------------------------------------------------- #
ROOT_MODEL = os.environ["RLM_ROOT_MODEL"]
SUB_MODEL = os.environ["RLM_SUB_MODEL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

root_lm = dspy.LM(ROOT_MODEL)
sub_lm = dspy.LM(SUB_MODEL)
embedding_lm = dspy.LM(EMBEDDING_MODEL)

dspy.configure(lm=root_lm)

# --------------------------------------------------------------------------- #
# In-process run store (replace with Postgres rlm_runs in production)
# --------------------------------------------------------------------------- #
class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Limits(BaseModel):
    max_iters: int = Field(default=12, ge=1, le=30)
    max_llm_calls: int = Field(default=24, ge=0, le=80)
    max_output_chars: int = Field(default=8_000, ge=500, le=20_000)
    timeout_s: int = Field(default=300, ge=10, le=900)


class CreateRun(BaseModel):
    tenant_id: str
    subject_id: str
    namespace: str = "default"
    context: str = Field(min_length=1, max_length=5_000_000)
    query: str = Field(min_length=1, max_length=20_000)
    limits: Limits = Field(default_factory=Limits)
    include_trajectory: bool = False


class WriteMemory(BaseModel):
    tenant_id: str
    subject_id: str
    namespace: str = "default"
    kind: str
    content: str = Field(min_length=1, max_length=12_000)
    metadata: dict = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0, le=1)
    expires_at: str | None = None


@dataclass
class Run:
    id: str
    status: str
    tenant_id: str
    subject_id: str
    created_at: float
    updated_at: float
    result: dict | None = None
    error: str | None = None
    trajectory: list[dict] | None = None
    recalled_memory_ids: list[str] | None = None


runs: dict[str, Run] = {}
memory_store: MemoryStore | None = None


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
@app.startup_handler
async def on_startup():
    global memory_store
    memory_store = await MemoryStore.create(embedding_lm)
    print("[startup] MemoryStore connected to Neon.")


@app.shutdown_handler
async def on_shutdown():
    if memory_store:
        await memory_store.close()
    print("[shutdown] MemoryStore pool closed.")


# --------------------------------------------------------------------------- #
# Background run executor
# --------------------------------------------------------------------------- #
def build_rlm(limits: Limits) -> dspy.RLM:
    return dspy.RLM(
        "context, query -> answer, evidence: list[str]",
        max_iters=limits.max_iters,
        max_llm_calls=limits.max_llm_calls,
        max_output_chars=limits.max_output_chars,
        sub_lm=sub_lm,
    )


async def _execute(run_id: str, payload: CreateRun) -> None:
    run = runs[run_id]
    run.status = RunStatus.RUNNING
    run.updated_at = time()

    try:
        rlm = build_rlm(payload.limits)

        prediction, recalled = await asyncio.wait_for(
            execute_run(
                run_id=run_id,
                tenant_id=payload.tenant_id,
                subject_id=payload.subject_id,
                namespace=payload.namespace,
                task=payload.query,
                corpus=payload.context,
                memory=memory_store,
                rlm=rlm,
            ),
            timeout=payload.limits.timeout_s,
        )

        run.result = {
            "answer": prediction.answer,
            "evidence": prediction.evidence,
        }
        run.recalled_memory_ids = [m.id for m in recalled]
        if payload.include_trajectory:
            run.trajectory = getattr(prediction, "trajectory", None)

        run.status = RunStatus.SUCCEEDED
        run.updated_at = time()

    except TimeoutError:
        run.status = RunStatus.FAILED
        run.error = f"Run exceeded {payload.limits.timeout_s}s"
        run.updated_at = time()

    except Exception as exc:
        run.status = RunStatus.FAILED
        run.error = f"{type(exc).__name__}: {exc}"
        run.updated_at = time()


# --------------------------------------------------------------------------- #
# Routes — RLM runs
# --------------------------------------------------------------------------- #
@app.post("/v1/rlm/runs")
async def create_run(request: Request):
    ok, err = require_auth(request)
    if not ok:
        return {"status_code": 401, "body": err}

    payload = CreateRun.model_validate_json(request.body)
    run_id = str(uuid.uuid4())
    now = time()

    runs[run_id] = Run(
        id=run_id,
        status=RunStatus.QUEUED,
        tenant_id=payload.tenant_id,
        subject_id=payload.subject_id,
        created_at=now,
        updated_at=now,
    )

    asyncio.create_task(_execute(run_id, payload))

    return {
        "status_code": 202,
        "headers": {"Location": f"/v1/rlm/runs/{run_id}"},
        "body": {"id": run_id, "status": RunStatus.QUEUED},
    }


@app.get("/v1/rlm/runs/:run_id")
async def get_run(request: Request):
    ok, err = require_auth(request)
    if not ok:
        return {"status_code": 401, "body": err}

    run_id = request.path_params["run_id"]
    run = runs.get(run_id)
    if not run:
        return {"status_code": 404, "body": {"error": "run_not_found"}}

    return {"status_code": 200, "body": asdict(run)}


# --------------------------------------------------------------------------- #
# Routes — Memory
# --------------------------------------------------------------------------- #
@app.post("/v1/memories")
async def write_memory(request: Request):
    ok, err = require_auth(request)
    if not ok:
        return {"status_code": 401, "body": err}

    payload = WriteMemory.model_validate_json(request.body)
    memory_id = await memory_store.write(**payload.model_dump())

    return {"status_code": 201, "body": {"id": memory_id, "status": "stored"}}


@app.get("/v1/memories/search")
async def search_memory(request: Request):
    ok, err = require_auth(request)
    if not ok:
        return {"status_code": 401, "body": err}

    q = request.query_params
    results = await memory_store.search(
        tenant_id=q["tenant_id"],
        subject_id=q["subject_id"],
        namespace=q.get("namespace", "default"),
        query=q["q"],
        limit=min(int(q.get("limit", 8)), 20),
    )

    return {
        "status_code": 200,
        "body": {
            "items": [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "content": item.content,
                    "metadata": item.metadata,
                    "score": item.score,
                }
                for item in results
            ]
        },
    }


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/healthz")
async def healthz(request: Request):
    return {"status_code": 200, "body": {"ok": True}}


if __name__ == "__main__":
    app.start(port=8080)
