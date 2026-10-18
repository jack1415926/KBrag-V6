"""Per-request identity, deadline, and cooperative cancellation context."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


class RequestCancelled(RuntimeError):
    """Raised when an API request is cancelled or reaches its deadline."""


_QID: ContextVar[str | None] = ContextVar("kbrag_qid", default=None)
_DEADLINE: ContextVar[float | None] = ContextVar("kbrag_deadline", default=None)
_CANCEL_EVENT: ContextVar[threading.Event | None] = ContextVar("kbrag_cancel_event", default=None)


@contextmanager
def request_scope(
    *,
    qid: str | int | None = None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> Iterator[None]:
    """Set request-local state and restore the previous context on exit."""
    tokens = (
        (_QID, _QID.set(None if qid is None else str(qid))),
        (_DEADLINE, _DEADLINE.set(deadline)),
        (_CANCEL_EVENT, _CANCEL_EVENT.set(cancel_event)),
    )
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def set_current_qid(qid: str | int | None) -> None:
    """Update only the current request context's question identifier."""
    _QID.set(None if qid is None else str(qid))


def get_current_qid() -> str | None:
    """Return the question identifier isolated to the current context."""
    return _QID.get()


def has_deadline() -> bool:
    """Return whether the current context has an API request deadline."""
    return _DEADLINE.get() is not None


def check_cancelled() -> None:
    """Raise when cancellation was requested or the deadline has elapsed."""
    event = _CANCEL_EVENT.get()
    if event is not None and event.is_set():
        raise RequestCancelled("request cancelled")
    deadline = _DEADLINE.get()
    if deadline is not None and time.monotonic() >= deadline:
        raise RequestCancelled("request deadline exceeded")


def remaining_timeout(default: float) -> float:
    """Cap a downstream timeout to the request's remaining wall-clock budget."""
    check_cancelled()
    deadline = _DEADLINE.get()
    if deadline is None:
        return float(default)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RequestCancelled("request deadline exceeded")
    return max(0.001, min(float(default), remaining))


def sleep_with_cancel(seconds: float) -> None:
    """Wait for retry backoff while allowing a cancellation event to wake it."""
    timeout = remaining_timeout(seconds)
    event = _CANCEL_EVENT.get()
    if event is None:
        time.sleep(timeout)
    elif event.wait(timeout):
        raise RequestCancelled("request cancelled")
    check_cancelled()
