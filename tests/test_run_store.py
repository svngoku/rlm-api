from __future__ import annotations

import pytest

from run_store import RunStatus, failure_status, retry_delay_seconds


def test_retry_backoff_is_exponential_and_bounded() -> None:
    assert [retry_delay_seconds(attempt) for attempt in range(1, 5)] == [
        1,
        2,
        4,
        8,
    ]
    assert retry_delay_seconds(50) == 300
    with pytest.raises(ValueError):
        retry_delay_seconds(0)


def test_failure_transition_retries_until_last_attempt() -> None:
    assert failure_status(1, 3) is RunStatus.QUEUED
    assert failure_status(2, 3) is RunStatus.QUEUED
    assert failure_status(3, 3) is RunStatus.FAILED
    with pytest.raises(ValueError):
        failure_status(4, 3)
