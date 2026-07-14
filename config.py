"""Validated runtime configuration."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass


class ConfigurationError(ValueError):
    """Raised when deployment configuration is unsafe or incomplete."""


@dataclass(frozen=True)
class Credential:
    tenant_id: str
    subject_id: str | None = None
    role: str = "client"


@dataclass(frozen=True)
class Settings:
    database_url: str
    root_model: str
    sub_model: str
    embedding_model: str
    embedding_dim: int
    api_keys: Mapping[str, Credential]
    db_pool_size: int = 10
    embedded_worker: bool = False
    worker_poll_seconds: float = 1.0
    worker_stale_seconds: int = 600
    worker_batch_size: int = 1
    worker_max_poll_failures: int = 5

    @property
    def public_model_config(self) -> dict[str, str | int]:
        return {
            "root_model": self.root_model,
            "sub_model": self.sub_model,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def required(name: str) -> str:
            value = env.get(name, "").strip()
            if not value:
                raise ConfigurationError(f"{name} must be set to a non-empty value")
            return value

        database_url = required("DATABASE_URL")
        root_model = required("RLM_ROOT_MODEL")
        sub_model = required("RLM_SUB_MODEL")
        embedding_model = required("EMBEDDING_MODEL")
        api_keys = _parse_api_keys(required("API_KEYS_JSON"))
        embedding_dim = _positive_int(env, "EMBEDDING_DIM")
        db_pool_size = _positive_int(env, "DB_POOL_SIZE", default=10)
        worker_stale_seconds = _positive_int(env, "WORKER_STALE_SECONDS", default=600)
        worker_batch_size = _positive_int(env, "WORKER_BATCH_SIZE", default=1)
        worker_max_poll_failures = _positive_int(env, "WORKER_MAX_POLL_FAILURES", default=5)
        worker_poll_seconds = _positive_float(env, "WORKER_POLL_SECONDS", default=1.0)
        if root_model == sub_model:
            warnings.warn(
                "RLM_ROOT_MODEL and RLM_SUB_MODEL are identical; this is allowed "
                "but may remove the intended root/sub-model specialization.",
                stacklevel=2,
            )
        return cls(
            database_url=database_url,
            root_model=root_model,
            sub_model=sub_model,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            api_keys=api_keys,
            db_pool_size=db_pool_size,
            embedded_worker=_bool(env.get("EMBEDDED_WORKER", "false")),
            worker_poll_seconds=worker_poll_seconds,
            worker_stale_seconds=worker_stale_seconds,
            worker_batch_size=worker_batch_size,
            worker_max_poll_failures=worker_max_poll_failures,
        )


def _parse_api_keys(raw: str) -> dict[str, Credential]:
    try:
        decoded: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("API_KEYS_JSON must be valid JSON") from exc
    if not isinstance(decoded, dict) or not decoded:
        raise ConfigurationError("API_KEYS_JSON must be a non-empty JSON object")

    credentials: dict[str, Credential] = {}
    for token, value in decoded.items():
        if not isinstance(token, str) or not token.strip():
            raise ConfigurationError("API_KEYS_JSON contains an empty API key")
        if isinstance(value, str):
            tenant_id = value.strip()
            subject_id = None
            role = "client"
        elif isinstance(value, dict):
            tenant = value.get("tenant_id")
            subject = value.get("subject_id")
            configured_role = value.get("role", "client")
            tenant_id = tenant.strip() if isinstance(tenant, str) else ""
            subject_id = subject.strip() if isinstance(subject, str) else None
            role = configured_role.strip() if isinstance(configured_role, str) else "client"
        else:
            raise ConfigurationError(
                "API_KEYS_JSON values must be tenant strings or credential objects"
            )
        if not tenant_id:
            raise ConfigurationError("Every API key must map to a tenant_id")
        credentials[token] = Credential(tenant_id=tenant_id, subject_id=subject_id, role=role)
    return credentials


def _positive_int(env: Mapping[str, str], name: str, *, default: int | None = None) -> int:
    raw = env.get(name)
    if raw is None and default is not None:
        return default
    try:
        value = int(raw or "")
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be a positive integer")
    return value


def _positive_float(env: Mapping[str, str], name: str, *, default: float) -> float:
    try:
        value = float(env.get(name, str(default)))
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be positive") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError("EMBEDDED_WORKER must be a boolean")
