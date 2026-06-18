"""Session-only history helpers for the dashboard."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

try:  # pragma: no cover - streamlit is optional in the test environment
    import streamlit as st
except ImportError:  # pragma: no cover
    class _FallbackState(dict):
        pass

    class _FallbackStreamlit:
        session_state: _FallbackState = _FallbackState()

    st = _FallbackStreamlit()  # type: ignore[assignment]


_HISTORY_KEY = "gecco_mh_dashboard_history"


def get_history() -> list[dict[str, Any]]:
    """Return the current in-session history list."""

    return st.session_state.setdefault(_HISTORY_KEY, [])


def append_snapshot(snapshot: dict[str, Any], *, max_points: int | None = None) -> list[dict[str, Any]]:
    """Append a copy of *snapshot* to the session history."""

    history = get_history()
    entry = deepcopy(snapshot)
    entry.setdefault("captured_at", datetime.now(timezone.utc).isoformat())
    history.append(entry)
    if max_points is not None and max_points > 0 and len(history) > max_points:
        del history[:-max_points]
    return history


def latest_snapshot() -> dict[str, Any] | None:
    history = get_history()
    return history[-1] if history else None


def clear_history() -> None:
    st.session_state[_HISTORY_KEY] = []
