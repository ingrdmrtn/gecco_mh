"""Reusable dashboard foundation helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd

try:  # pragma: no cover - streamlit is optional in the test environment
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None


_STATUS_METADATA: dict[str, dict[str, Any]] = {
    "running": {"status": "running", "label": "Running", "tone": "info", "icon": "⏳", "terminal": False},
    "complete": {"status": "complete", "label": "Complete", "tone": "success", "icon": "✅", "terminal": True},
    "complete_no_success": {
        "status": "complete_no_success",
        "label": "Complete (no success)",
        "tone": "success",
        "icon": "✅",
        "terminal": True,
    },
    "success": {"status": "success", "label": "Success", "tone": "success", "icon": "✅", "terminal": True},
    "recovery_failed": {
        "status": "recovery_failed",
        "label": "Recovery failed",
        "tone": "error",
        "icon": "⚠️",
        "terminal": True,
    },
    "failed": {"status": "failed", "label": "Failed", "tone": "error", "icon": "⛔", "terminal": True},
    "validation_error": {
        "status": "validation_error",
        "label": "Validation error",
        "tone": "error",
        "icon": "⛔",
        "terminal": True,
    },
    "fit_error": {"status": "fit_error", "label": "Fit error", "tone": "error", "icon": "⛔", "terminal": True},
    "error": {"status": "error", "label": "Error", "tone": "error", "icon": "⛔", "terminal": True},
}

_NEUTRAL_STATUS = {"status": "unknown", "label": "Unknown", "tone": "neutral", "icon": "•", "terminal": False}


def normalize_status(status: str | None) -> str:
    """Return a canonical lowercase status token."""

    if status is None:
        return "unknown"
    token = str(status).strip().lower().replace("-", "_").replace(" ", "_")
    return token or "unknown"


def status_metadata(status: str | None) -> dict[str, Any]:
    """Return neutral metadata for known and unknown statuses."""

    token = normalize_status(status)
    return dict(_STATUS_METADATA.get(token, _NEUTRAL_STATUS))


def format_value(value: Any, *, default: str = "—", precision: int = 2) -> str:
    """Format common dashboard values without introducing locale surprises."""

    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        rendered = f"{value:.{precision}f}".rstrip("0").rstrip(".")
        return rendered or "0"
    text = str(value).strip()
    return text or default


@contextmanager
def sidebar_section(title: str) -> Iterator[None]:
    """Render a lightweight grouped sidebar section."""

    if st is None:
        yield
        return

    sidebar_header(title)
    yield


def sidebar_header(title: str, *, caption: str | None = None) -> None:
    """Render a sidebar heading with optional descriptive caption."""

    if st is None:
        return
    st.sidebar.header(title)
    if caption:
        st.sidebar.caption(caption)


def section_header(title: str, *, caption: str | None = None) -> None:
    """Render a standard section header."""

    if st is None:
        return
    st.subheader(title)
    if caption:
        st.caption(caption)


def empty_state(message: str) -> None:
    """Render a neutral empty state."""

    if st is None:
        return
    st.info(message)


def key_value(label: str, value: Any) -> None:
    """Render a compact key/value line."""

    if st is None:
        return
    st.write(label, value)


def metric_grid(items: list[dict[str, Any]]) -> None:
    """Render a responsive row of Streamlit metrics."""

    if st is None or not items:
        return

    columns = st.columns(len(items)) if hasattr(st, "columns") else []

    def _render_metric(item: dict[str, Any]) -> None:
        try:
            st.metric(item["label"], item["value"], delta=item.get("delta"), help=item.get("help"))
        except TypeError:
            st.metric(item["label"], item["value"])

    for index, item in enumerate(items):
        target = columns[index] if index < len(columns) else None
        if target is not None:
            with target:
                _render_metric(item)
        else:
            _render_metric(item)


def write_dataframe(frame: pd.DataFrame | None, *, empty_message: str = "No rows available.") -> None:
    """Render a DataFrame with a stable empty-state fallback."""

    if st is None or frame is None:
        return
    if frame.empty:
        st.caption(empty_message)
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


def debug_details(label: str, body: Any) -> None:
    """Render hidden debug material behind a collapsed expander."""

    if st is None:
        return

    with st.expander(label, expanded=False):
        if isinstance(body, (dict, list, tuple)):
            st.json(body)
        else:
            st.code(format_value(body), language="text")
