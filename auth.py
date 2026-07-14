"""Bearer token middleware for Robyn."""
from __future__ import annotations

import os

from robyn import Request

_SECRET = os.getenv("API_SECRET_KEY", "")

PUBLIC_PATHS = {"/healthz"}


def require_auth(request: Request) -> tuple[bool, dict]:
    """Return (authorized, error_body). Call before handler logic."""
    if request.url.path in PUBLIC_PATHS:
        return True, {}

    if not _SECRET:
        # Auth disabled — only safe inside a private network.
        return True, {}

    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return False, {"error": "missing_bearer_token"}

    token = auth.removeprefix("Bearer ").strip()
    if token != _SECRET:
        return False, {"error": "invalid_token"}

    return True, {}
