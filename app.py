"""Robyn API for durable DSPy Recursive Language Model runs."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from robyn import Request, Response, Robyn

from auth import (
    AuthenticationError,
    Principal,
    authenticate,
    enforce_scope,
)
from config import Settings
from logging_config import configure_logging
from logging_config import context as log_context
from memory import JSONValue, MemoryKind, MemoryStore
from model_runtime import DSPyEmbeddingProvider
from run_store import RunStore
from worker import run_worker_loop

load_dotenv()
configure_logging()
settings = Settings.from_env()
logger = logging.getLogger("rlm.api")
app = Robyn(__file__)

run_store: RunStore | None = None
memory_store: MemoryStore | None = None
embedded_stop: asyncio.Event | None = None
embedded_task: asyncio.Task[None] | None = None


class Limits(BaseModel):
    max_iters: int = Field(default=12, ge=1, le=30)
    max_llm_calls: int = Field(default=24, ge=0, le=80)
    max_output_chars: int = Field(default=8_000, ge=500, le=20_000)
    timeout_s: int = Field(default=300, ge=10, le=900)


class CreateRun(BaseModel):
    tenant_id: str | None = None
    subject_id: str = Field(min_length=1, max_length=256)
    namespace: str = Field(default="default", min_length=1, max_length=128)
    context: str = Field(min_length=1, max_length=5_000_000)
    query: str = Field(min_length=1, max_length=20_000)
    limits: Limits = Field(default_factory=Limits)
    include_trajectory: bool = False
    max_attempts: int = Field(default=3, ge=1, le=10)


class WriteMemory(BaseModel):
    tenant_id: str | None = None
    subject_id: str = Field(min_length=1, max_length=256)
    namespace: str = Field(default="default", min_length=1, max_length=128)
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=12_000)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0, le=1)
    expires_at: datetime | None = None


@app.startup_handler
async def on_startup() -> None:
    global run_store, memory_store, embedded_stop, embedded_task
    queue_store = await RunStore.create(
        settings.database_url,
        pool_size=settings.db_pool_size,
        application_name="rlm-api-queue",
    )
    try:
        memories = await MemoryStore.create(
            database_url=settings.database_url,
            pool_size=settings.db_pool_size,
            embedding_provider=DSPyEmbeddingProvider(settings.embedding_model),
            embedding_dim=settings.embedding_dim,
            application_name="rlm-api-memory",
        )
    except Exception:
        await queue_store.close()
        raise
    run_store = queue_store
    memory_store = memories
    if settings.embedded_worker:
        embedded_stop = asyncio.Event()
        embedded_task = asyncio.create_task(
            run_worker_loop(
                settings,
                store=run_store,
                memory=memory_store,
                stop_event=embedded_stop,
            )
        )
    logger.info(
        "api_started",
        extra={"context": {"embedded_worker": settings.embedded_worker}},
    )


@app.shutdown_handler
async def on_shutdown() -> None:
    if embedded_stop is not None:
        embedded_stop.set()
    if embedded_task is not None:
        await asyncio.gather(embedded_task, return_exceptions=True)
    if memory_store is not None:
        await memory_store.close()
    if run_store is not None:
        await run_store.close()
    logger.info("api_stopped")


@app.post("/v1/rlm/runs")
async def create_run(request: Request) -> Response:
    request_id = _request_id(request)
    principal, error = _principal(request, request_id)
    if error is not None:
        return error
    assert principal is not None
    try:
        payload = CreateRun.model_validate_json(request.body)
        enforce_scope(
            principal,
            tenant_id=payload.tenant_id,
            subject_id=payload.subject_id,
        )
    except ValidationError as exc:
        return _response(
            422,
            {"error": "invalid_request", "details": _validation_details(exc)},
            request_id,
        )
    except PermissionError as exc:
        return _response(403, {"error": str(exc)}, request_id)
    if run_store is None:
        return _response(503, {"error": "service_not_ready"}, request_id)
    try:
        run_id = await run_store.enqueue(
            tenant_id=principal.tenant_id,
            subject_id=payload.subject_id,
            namespace=payload.namespace,
            task=payload.query,
            context=payload.context,
            limits=payload.limits.model_dump(),
            model_config=settings.public_model_config,
            include_trajectory=payload.include_trajectory,
            max_attempts=payload.max_attempts,
        )
    except Exception as error:
        logger.error(
            "run_enqueue_failed",
            extra={
                "context": log_context(
                    request_id=request_id,
                    tenant_id=principal.tenant_id,
                    exception_type=type(error).__name__,
                )
            },
        )
        return _response(503, {"error": "queue_unavailable"}, request_id)
    logger.info(
        "run_enqueued",
        extra={
            "context": log_context(
                request_id=request_id,
                run_id=run_id,
                tenant_id=principal.tenant_id,
            )
        },
    )
    return _response(
        202,
        {"id": run_id, "status": "queued"},
        request_id,
        {"Location": f"/v1/rlm/runs/{run_id}"},
    )


@app.get("/v1/rlm/runs/:run_id")
async def get_run(request: Request) -> Response:
    request_id = _request_id(request)
    principal, error = _principal(request, request_id)
    if error is not None:
        return error
    assert principal is not None
    if run_store is None:
        return _response(503, {"error": "service_not_ready"}, request_id)
    run_id = request.path_params.get("run_id", "")
    try:
        run = await run_store.get(run_id=run_id, tenant_id=principal.tenant_id)
    except Exception as error:
        logger.error(
            "run_read_failed",
            extra={
                "context": log_context(
                    request_id=request_id,
                    run_id=run_id,
                    tenant_id=principal.tenant_id,
                    exception_type=type(error).__name__,
                )
            },
        )
        return _response(503, {"error": "database_unavailable"}, request_id)
    if run is None:
        return _response(404, {"error": "run_not_found"}, request_id)
    try:
        enforce_scope(principal, subject_id=str(run["subject_id"]))
    except PermissionError as exc:
        return _response(403, {"error": str(exc)}, request_id)
    return _response(200, run, request_id)


@app.post("/v1/memories")
async def write_memory(request: Request) -> Response:
    request_id = _request_id(request)
    principal, error = _principal(request, request_id)
    if error is not None:
        return error
    assert principal is not None
    try:
        payload = WriteMemory.model_validate_json(request.body)
        enforce_scope(
            principal,
            tenant_id=payload.tenant_id,
            subject_id=payload.subject_id,
        )
    except ValidationError as exc:
        return _response(
            422,
            {"error": "invalid_request", "details": _validation_details(exc)},
            request_id,
        )
    except PermissionError as exc:
        return _response(403, {"error": str(exc)}, request_id)
    if memory_store is None:
        return _response(503, {"error": "service_not_ready"}, request_id)
    try:
        memory_id = await memory_store.write(
            tenant_id=principal.tenant_id,
            subject_id=payload.subject_id,
            namespace=payload.namespace,
            kind=payload.kind,
            content=payload.content,
            metadata=payload.metadata,
            importance=payload.importance,
            expires_at=payload.expires_at.isoformat() if payload.expires_at else None,
        )
    except Exception as error:
        logger.error(
            "memory_write_failed",
            extra={
                "context": log_context(
                    request_id=request_id,
                    tenant_id=principal.tenant_id,
                    exception_type=type(error).__name__,
                )
            },
        )
        return _response(503, {"error": "memory_store_unavailable"}, request_id)
    logger.info(
        "memory_stored",
        extra={
            "context": log_context(
                request_id=request_id,
                tenant_id=principal.tenant_id,
                memory_id=memory_id,
            )
        },
    )
    return _response(201, {"id": memory_id, "status": "stored"}, request_id)


@app.get("/v1/memories/search")
async def search_memory(request: Request) -> Response:
    request_id = _request_id(request)
    principal, error = _principal(request, request_id)
    if error is not None:
        return error
    assert principal is not None
    query = request.query_params
    subject_id = query.get("subject_id", "")
    search_text = query.get("q", "")
    if not subject_id or not search_text:
        return _response(400, {"error": "subject_id_and_q_are_required"}, request_id)
    try:
        enforce_scope(
            principal,
            tenant_id=query.get("tenant_id"),
            subject_id=subject_id,
        )
        limit = int(query.get("limit", "8"))
        if not 1 <= limit <= 20:
            raise ValueError
    except PermissionError as exc:
        return _response(403, {"error": str(exc)}, request_id)
    except ValueError:
        return _response(400, {"error": "limit_must_be_between_1_and_20"}, request_id)
    if memory_store is None:
        return _response(503, {"error": "service_not_ready"}, request_id)
    try:
        results = await memory_store.search(
            tenant_id=principal.tenant_id,
            subject_id=subject_id,
            namespace=query.get("namespace", "default"),
            query=search_text,
            limit=limit,
        )
    except Exception as error:
        logger.error(
            "memory_search_failed",
            extra={
                "context": log_context(
                    request_id=request_id,
                    tenant_id=principal.tenant_id,
                    exception_type=type(error).__name__,
                )
            },
        )
        return _response(503, {"error": "memory_store_unavailable"}, request_id)
    logger.info(
        "memory_search_completed",
        extra={
            "context": log_context(
                request_id=request_id,
                tenant_id=principal.tenant_id,
                result_count=len(results),
            )
        },
    )
    return _response(
        200,
        {
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
        request_id,
    )


@app.get("/v1/models")
async def models(request: Request) -> Response:
    request_id = _request_id(request)
    _, error = _principal(request, request_id)
    if error is not None:
        return error
    return _response(200, settings.public_model_config, request_id)


@app.get("/livez")
async def livez(request: Request) -> Response:
    return _response(200, {"ok": True}, _request_id(request))


@app.get("/healthz")
async def healthz(request: Request) -> Response:
    request_id = _request_id(request)
    worker_ready = embedded_task is None or not embedded_task.done()
    database_ready = run_store is not None and await run_store.ready()
    ready = database_ready and worker_ready
    return _response(
        200 if ready else 503,
        {
            "ok": ready,
            "database": "ready" if database_ready else "unavailable",
            "embedded_worker": "ready" if worker_ready else "terminated",
        },
        request_id,
    )


def _principal(request: Request, request_id: str) -> tuple[Principal | None, Response | None]:
    try:
        return authenticate(request, settings.api_keys), None
    except AuthenticationError as exc:
        return None, _response(401, {"error": exc.code}, request_id)


def _request_id(request: Request) -> str:
    for key, value in request.headers.items():
        if key.lower() != "x-request-id":
            continue
        if 0 < len(value) <= 128:
            safe = value.isascii() and all(
                character.isalnum() or character in "-_.:" for character in value
            )
            if safe:
                return value
    return str(uuid.uuid4())


def _response(
    status_code: int,
    body: object,
    request_id: str,
    headers: dict[str, str] | None = None,
) -> Response:
    response_headers = {
        "Content-Type": "application/json",
        "X-Request-ID": request_id,
        **(headers or {}),
    }
    return Response(
        status_code,
        headers=response_headers,
        body=json.dumps(body, separators=(",", ":")),
    )


def _validation_details(error: ValidationError) -> list[dict[str, object]]:
    return [
        {
            "location": list(item["loc"]),
            "message": item["msg"],
            "type": item["type"],
        }
        for item in error.errors()
    ]


if __name__ == "__main__":
    app.start(port=8080)
