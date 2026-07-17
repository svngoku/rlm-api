"""Fail-closed, tenant-bound bearer authentication."""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from config import Credential


class RequestLike(Protocol):
    headers: Mapping[str, str]


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    subject_id: str | None
    role: str


class AuthenticationError(ValueError):
    """An intentionally client-safe authentication failure."""

    def __init__(self, code: str = "invalid_bearer_token") -> None:
        super().__init__(code)
        self.code = code


def authenticate(request: RequestLike, credentials: Mapping[str, Credential]) -> Principal:
    """Authenticate without data-dependent short-circuiting across configured keys."""
    header = _header(request.headers, "authorization")
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("missing_bearer_token")
    supplied = token.strip()

    matched: Credential | None = None
    for configured_token, credential in credentials.items():
        if hmac.compare_digest(supplied, configured_token):
            matched = credential
    if matched is None:
        raise AuthenticationError()
    return Principal(
        tenant_id=matched.tenant_id,
        subject_id=matched.subject_id,
        role=matched.role,
    )


def enforce_scope(
    principal: Principal,
    *,
    tenant_id: str | None = None,
    subject_id: str | None = None,
) -> None:
    if tenant_id is not None and not hmac.compare_digest(tenant_id, principal.tenant_id):
        raise PermissionError("tenant_mismatch")
    if (
        principal.subject_id is not None
        and subject_id is not None
        and not hmac.compare_digest(subject_id, principal.subject_id)
    ):
        raise PermissionError("subject_mismatch")


def _header(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""
