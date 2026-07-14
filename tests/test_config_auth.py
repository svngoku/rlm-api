from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from auth import AuthenticationError, authenticate, enforce_scope
from config import ConfigurationError, Credential, Settings


@dataclass
class FakeRequest:
    headers: dict[str, str]


def valid_env() -> dict[str, str]:
    return {
        "DATABASE_URL": "postgresql://example.invalid/db",
        "RLM_ROOT_MODEL": "provider/root-selected-by-operator",
        "RLM_SUB_MODEL": "provider/sub-selected-by-operator",
        "EMBEDDING_MODEL": "provider/embedding-selected-by-operator",
        "EMBEDDING_DIM": "1536",
        "API_KEYS_JSON": json.dumps({"opaque-secret": "tenant-a"}),
    }


def test_settings_require_explicit_models_and_credentials() -> None:
    env = valid_env()
    del env["RLM_ROOT_MODEL"]
    with pytest.raises(ConfigurationError, match="RLM_ROOT_MODEL"):
        Settings.from_env(env)

    env = valid_env()
    env["API_KEYS_JSON"] = "{}"
    with pytest.raises(ConfigurationError, match="non-empty"):
        Settings.from_env(env)

    env = valid_env()
    env["WORKER_MAX_POLL_FAILURES"] = "0"
    with pytest.raises(ConfigurationError, match="WORKER_MAX_POLL_FAILURES"):
        Settings.from_env(env)

    env = valid_env()
    env["API_KEYS_JSON"] = json.dumps({" padded-key ": "tenant-a"})
    with pytest.raises(ConfigurationError, match="surrounding whitespace"):
        Settings.from_env(env)


def test_settings_parse_tenant_and_subject_binding() -> None:
    env = valid_env()
    env["API_KEYS_JSON"] = json.dumps(
        {
            "key": {
                "tenant_id": "tenant-a",
                "subject_id": "subject-a",
                "role": "writer",
            }
        }
    )
    settings = Settings.from_env(env)
    assert settings.embedding_dim == 1536
    assert settings.worker_max_poll_failures == 5
    assert settings.worker_recovery_batch_size == 100
    assert settings.api_keys["key"].subject_id == "subject-a"


def test_authentication_is_fail_closed_and_tenant_bound() -> None:
    credentials = {"secret": Credential("tenant-a", "subject-a")}
    with pytest.raises(AuthenticationError):
        authenticate(FakeRequest({}), credentials)
    with pytest.raises(AuthenticationError):
        authenticate(FakeRequest({"Authorization": "Bearer wrong"}), credentials)

    principal = authenticate(FakeRequest({"Authorization": "Bearer secret"}), credentials)
    enforce_scope(principal, tenant_id="tenant-a", subject_id="subject-a")
    with pytest.raises(PermissionError, match="tenant_mismatch"):
        enforce_scope(principal, tenant_id="tenant-b")
    with pytest.raises(PermissionError, match="subject_mismatch"):
        enforce_scope(principal, subject_id="subject-b")
