"""Central Streamlit theme injection for the dashboard."""

from __future__ import annotations

try:  # pragma: no cover - streamlit is optional in the test environment
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None


DASHBOARD_THEME_CSS = """
<style>
  .block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.8rem;
  }

  [data-testid="stSidebar"] {
    border-right: 1px solid rgba(49, 51, 63, 0.14);
  }

  [data-testid="stMetric"] {
    background: rgba(49, 51, 63, 0.04);
    border: 1px solid rgba(49, 51, 63, 0.08);
    border-radius: 0.75rem;
    padding: 0.6rem 0.85rem;
  }

  [data-testid="stExpander"] details {
    border-radius: 0.75rem;
  }

  code {
    white-space: pre-wrap;
  }
</style>
"""


def apply_dashboard_theme() -> None:
    """Inject the shared dashboard theme."""

    if st is None:
        return
    st.markdown(DASHBOARD_THEME_CSS, unsafe_allow_html=True)
