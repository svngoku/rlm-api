"""Small structured-JSON logging adapter with context fields."""
from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, Mapping):
            payload.update(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


def context(
    *,
    request_id: str | None = None,
    run_id: str | None = None,
    tenant_id: str | None = None,
    **fields: object,
) -> dict[str, object]:
    values: dict[str, object] = dict(fields)
    if request_id:
        values["request_id"] = request_id
    if run_id:
        values["run_id"] = run_id
    if tenant_id:
        values["tenant_id"] = tenant_id
    return values
